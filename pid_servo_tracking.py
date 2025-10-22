"""
Simple AI Camera Servo Tracking with Live Visualization
Shows what the PID controller is doing in real-time!

PID CONTROL OVERVIEW:
====================
A PID controller continuously calculates an "error" (difference between 
desired position and current position) and applies corrections based on:
- P (Proportional): How far off you are RIGHT NOW
- I (Integral): How long you've been off over TIME  
- D (Derivative): How fast you're changing

Think of it like driving a car:
- P: "I'm 10 feet from the parking spot, turn more!"
- I: "I've been drifting left for 5 seconds, compensate!"
- D: "I'm turning too fast, slow down the steering!"
"""
import logging
import time
from typing import Protocol
import numpy as np
import cv2
from pydantic import BaseModel, Field, ConfigDict, model_validator

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Models
# ============================================================================

class PIDGainsConfig(BaseModel):
    """
    PID Controller Gains - These are the "tuning knobs" for your controller.
    
    Think of it like adjusting a shower:
    - kp: How quickly you react to temperature changes
    - ki: How much you compensate for long-term drift
    - kd: How much you smooth out rapid changes
    """
    model_config = ConfigDict(frozen=True)
    
    kp: float = Field(
        default=3.0, 
        gt=0,
        description="Proportional gain - Reaction strength to current error. "
                    "Higher = faster response but more oscillation/overshoot. "
                    "Start with 1.0 and increase until it responds quickly."
    )
    
    ki: float = Field(
        default=0.2, 
        ge=0,
        description="Integral gain - Corrects accumulated error over time. "
                    "Eliminates steady-state error (being 'stuck' slightly off-target). "
                    "Too high causes instability. Start at 0.0, add small amounts."
    )
    
    kd: float = Field(
        default=0.8, 
        ge=0,
        description="Derivative gain - Dampens rapid changes, reduces overshoot. "
                    "Like power steering damping. Prevents oscillation. "
                    "Start at 0.0, increase to reduce wobbling."
    )


class ServoConfig(BaseModel):
    """Servo configuration."""
    model_config = ConfigDict(frozen=True)
    
    i2c_address: int = Field(default=0x40, description="PCA9685 I2C address")
    pan_channel: int = Field(default=0, description="Pan servo PWM channel (0-15)")
    tilt_channel: int = Field(default=1, description="Tilt servo PWM channel (0-15)")
    pwm_freq: int = Field(default=50, description="PWM frequency in Hz (50Hz = 20ms period)")
    
    # These need calibration for your specific servos!
    pulse_min: int = Field(
        default=150, 
        description="PWM pulse for 0° (calibrate by testing your servo)"
    )
    pulse_max: int = Field(
        default=600, 
        description="PWM pulse for 180° (calibrate by testing your servo)"
    )
    
    # Physical servo angle limits (degrees)
    pan_min: float = Field(default=0.0, description="Minimum pan angle")
    pan_max: float = Field(default=180.0, description="Maximum pan angle")
    tilt_min: float = Field(default=0.0, description="Minimum tilt angle")
    tilt_max: float = Field(default=180.0, description="Maximum tilt angle")
    
    # Center/neutral positions (where servos point straight ahead)
    pan_center: float = Field(default=90.0, description="Pan center position")
    tilt_center: float = Field(default=90.0, description="Tilt center position")
    
    # Camera field of view - used to convert pixel position to angle
    fov_horizontal: float = Field(
        default=78.3, 
        description="Camera horizontal FOV in degrees (AI Camera = 78.3°)"
    )
    fov_vertical: float = Field(
        default=58.7, 
        description="Camera vertical FOV in degrees (AI Camera = 58.7°)"
    )


class SystemConfig(BaseModel):
    """System configuration."""
    model_config = ConfigDict(frozen=True)
    
    servo: ServoConfig = Field(default_factory=ServoConfig)
    pan_pid: PIDGainsConfig = Field(default_factory=PIDGainsConfig)
    tilt_pid: PIDGainsConfig = Field(default_factory=PIDGainsConfig)
    
    camera_width: int = Field(default=640)
    camera_height: int = Field(default=480)
    camera_fps: int = Field(default=30)
    
    model_path: str = Field(
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    )
    confidence_threshold: float = Field(default=0.5)


# ============================================================================
# Detection Result
# ============================================================================

class Detection(BaseModel):
    """Simple detection result from AI Camera."""
    target_x: float = Field(ge=0.0, le=1.0, description="X position (0=left, 1=right)")
    target_y: float = Field(ge=0.0, le=1.0, description="Y position (0=top, 1=bottom)")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    class_name: str
    box: list[float] = Field(description="Bounding box [x, y, width, height]")


# ============================================================================
# PID Controller - The Heart of Smooth Servo Control
# ============================================================================

class PIDController:
    """
    PID Controller for smooth servo tracking.
    
    WHAT DOES IT DO?
    ================
    Imagine you're trying to keep a laser pointer on a moving target:
    - You see the target move left → you move the laser left
    - But you overshoot! Now it's too far left
    - You correct back right, but overshoot again...
    - This oscillation continues forever!
    
    PID prevents this by considering THREE things:
    
    1. PROPORTIONAL (P): "How far am I from target RIGHT NOW?"
       - Larger error = stronger correction
       - Problem: Can overshoot and oscillate
       - Like a spring pulling you toward the target
    
    2. INTEGRAL (I): "How LONG have I been off-target?"
       - Fixes steady-state error (being stuck slightly off)
       - Accumulates error over time, adds constant push
       - Problem: Can cause instability if too large
       - Like noticing you've been consistently too low and compensating
    
    3. DERIVATIVE (D): "How FAST am I approaching target?"
       - Slows down as you get close (dampening)
       - Reduces overshoot and oscillation
       - Problem: Sensitive to noise
       - Like brakes on a car
    
    FORMULA:
    ========
    output = (Kp × error) + (Ki × ∫error·dt) + (Kd × Δerror/dt)
    
    WHERE:
    - error = setpoint - measurement (how far off you are)
    - Kp, Ki, Kd are the "gains" (tuning knobs)
    
    TUNING GUIDE:
    =============
    Start with: Kp=1.0, Ki=0.0, Kd=0.0
    
    Step 1: Increase Kp until system responds quickly but oscillates slightly
    Step 2: Add Kd to reduce oscillation (start with Kp/4)
    Step 3: Add small Ki if there's steady-state error (start with Kp/15)
    
    Signs of bad tuning:
    - Oscillates/wobbles: Kp too high OR Kd too low
    - Slow response: Kp too low
    - Never quite reaches target: Add Ki
    - Unstable/diverges: Ki too high
    """
    
    def __init__(self, gains: PIDGainsConfig):
        self.gains = gains
        
        # State variables (reset between tracking sessions)
        self.integral = 0.0      # Accumulated error over time (I term)
        self.last_error = 0.0    # Previous error (for D term calculation)
        self.last_time: float | None = None  # Previous timestamp
        
        # For visualization
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output = 0.0
    
    def update(self, setpoint: float, measurement: float, current_time: float) -> float:
        """
        Calculate PID output (velocity command).
        
        Args:
            setpoint: Where you WANT to be (desired angle in degrees)
            measurement: Where you ARE now (current angle in degrees)
            current_time: Current timestamp (for calculating dt)
        
        Returns:
            velocity: How fast to move (degrees per second)
        
        Example:
            setpoint = 100°     (target is at 100°)
            measurement = 90°   (servo currently at 90°)
            → error = 10°       (need to move 10° right)
            → output = +15°/s   (move right at 15 degrees per second)
        """
        # Calculate error (positive = need to move in positive direction)
        error = setpoint - measurement
        
        # Calculate time delta (dt)
        if self.last_time is None:
            dt = 0.0  # First call, no delta yet
        else:
            dt = current_time - self.last_time
        self.last_time = current_time
        
        # No time passed? Return zero (avoid divide by zero)
        if dt <= 0:
            return 0.0
        
        # ================================================================
        # P TERM: Proportional - "How far off am I?"
        # ================================================================
        p_term = self.gains.kp * error
        
        # ================================================================
        # I TERM: Integral - "How long have I been off?"
        # ================================================================
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Anti-windup limits
        i_term = self.gains.ki * self.integral
        
        # ================================================================
        # D TERM: Derivative - "How fast am I changing?"
        # ================================================================
        d_term = self.gains.kd * (error - self.last_error) / dt
        self.last_error = error
        
        # ================================================================
        # COMBINE ALL TERMS
        # ================================================================
        output = p_term + i_term + d_term
        output = np.clip(output, -30.0, 30.0)  # Max ±30°/s
        
        # Store for visualization
        self.last_p_term = p_term
        self.last_i_term = i_term
        self.last_d_term = d_term
        self.last_output = output
        
        return float(output)
    
    def reset(self) -> None:
        """
        Reset PID state (call when starting new tracking session).
        
        Clears accumulated integral and error history.
        Important when switching targets or reinitializing.
        """
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_output = 0.0


# ============================================================================
# Servo Driver - Converts Angles to PWM Signals
# ============================================================================

class ServoDriver:
    """
    PCA9685 servo driver.
    
    HOW SERVOS WORK:
    ================
    Servos expect a PWM signal at 50Hz (one pulse every 20ms).
    The pulse width determines the angle:
    - 1.0ms pulse = 0° (typically)
    - 1.5ms pulse = 90° (center)
    - 2.0ms pulse = 180°
    
    PCA9685 has 12-bit resolution (0-4095) at 50Hz.
    We need to map angles to these pulse values.
    """
    
    def __init__(self, config: ServoConfig):
        self.config = config
        self.pca9685 = None
        self._setup_hardware()
    
    def _setup_hardware(self) -> None:
        """Initialize PCA9685 PWM driver."""
        try:
            from adafruit_pca9685 import PCA9685
            from board import SCL, SDA
            import busio
            
            i2c = busio.I2C(scl=SCL, sda=SDA)
            self.pca9685 = PCA9685(i2c_bus=i2c, address=self.config.i2c_address)
            self.pca9685.frequency = self.config.pwm_freq
            logger.info(f"PCA9685 initialized at 0x{self.config.i2c_address:02x}")
        except Exception as e:
            logger.error(f"Failed to initialize PCA9685: {e}")
            logger.warning("Running in simulation mode")
    
    def _angle_to_pulse(self, angle: float) -> int:
        """
        Convert servo angle (0-180°) to PWM pulse width.
        
        Linear mapping:
        - pulse_min (e.g., 150) = 0°
        - pulse_max (e.g., 600) = 180°
        
        Formula: pulse = min + (angle/180) × (max - min)
        
        Example with pulse_min=150, pulse_max=600:
        - 0° → 150
        - 90° → 375
        - 180° → 600
        """
        pulse = int(
            self.config.pulse_min + 
            (angle / 180.0) * (self.config.pulse_max - self.config.pulse_min)
        )
        return pulse
    
    def set_angles(self, pan: float, tilt: float) -> None:
        """
        Set both servo angles at once.
        
        Args:
            pan: Pan angle in degrees (0-180)
            tilt: Tilt angle in degrees (0-180)
        """
        pan_pulse = self._angle_to_pulse(angle=pan)
        tilt_pulse = self._angle_to_pulse(angle=tilt)
        
        if self.pca9685 is not None:
            # Set PWM duty cycle for each channel
            self.pca9685.channels[self.config.pan_channel].duty_cycle = pan_pulse
            self.pca9685.channels[self.config.tilt_channel].duty_cycle = tilt_pulse
        else:
            # Simulation mode (no hardware)
            logger.debug(f"[SIM] Pan={pan:.1f}° Tilt={tilt:.1f}°")
    
    def close(self) -> None:
        """Cleanup hardware."""
        if self.pca9685 is not None:
            self.pca9685.deinit()


# ============================================================================
# Visualizer - Shows what's happening
# ============================================================================

class TrackingVisualizer:
    """
    Real-time visualization of the tracking system.
    Shows detection, servo positions, and PID values.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.window_name = "AI Servo Tracker - PID Control Visualization"
        
        # Colors (BGR format for OpenCV)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
    
    def draw_annotations(
        self,
        frame: np.ndarray,
        detection: Detection | None,
        current_pan: float,
        current_tilt: float,
        desired_pan: float,
        desired_tilt: float,
        pan_pid: PIDController,
        tilt_pid: PIDController,
        fps: float
    ) -> np.ndarray:
        """
        Draw all annotations on the frame.
        
        Shows:
        - Bounding box around detected object
        - Crosshair at current servo aim point
        - Target marker where servos want to go
        - PID controller values
        - FPS and status info
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for info panels
        overlay = frame.copy()
        
        # ================================================================
        # Draw detection bounding box and target
        # ================================================================
        if detection is not None:
            # Bounding box
            x, y, box_w, box_h = detection.box
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + box_w) * w)
            y2 = int((y + box_h) * h)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_GREEN, 2)
            
            # Target center (where object is)
            target_x = int(detection.target_x * w)
            target_y = int(detection.target_y * h)
            
            # Draw target marker (circle + crosshair)
            cv2.circle(frame, (target_x, target_y), 10, self.COLOR_RED, 2)
            cv2.line(frame, (target_x - 15, target_y), (target_x + 15, target_y), self.COLOR_RED, 2)
            cv2.line(frame, (target_x, target_y - 15), (target_x, target_y + 15), self.COLOR_RED, 2)
            
            # Label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GREEN, 2)
        
        # ================================================================
        # Draw current aim point (where servos are pointing)
        # ================================================================
        # Convert servo angles back to screen coordinates
        pan_offset_norm = (current_pan - self.config.servo.pan_center) / self.config.servo.fov_horizontal
        tilt_offset_norm = -(current_tilt - self.config.servo.tilt_center) / self.config.servo.fov_vertical
        
        current_x = int((0.5 + pan_offset_norm) * w)
        current_y = int((0.5 + tilt_offset_norm) * h)
        
        # Draw cyan crosshair (current position)
        cv2.drawMarker(frame, (current_x, current_y), self.COLOR_CYAN, 
                      cv2.MARKER_CROSS, 30, 2)
        
        # ================================================================
        # Draw desired aim point (where servos want to go)
        # ================================================================
        if detection is not None:
            desired_pan_offset_norm = (desired_pan - self.config.servo.pan_center) / self.config.servo.fov_horizontal
            desired_tilt_offset_norm = -(desired_tilt - self.config.servo.tilt_center) / self.config.servo.fov_vertical
            
            desired_x = int((0.5 + desired_pan_offset_norm) * w)
            desired_y = int((0.5 + desired_tilt_offset_norm) * h)
            
            # Draw yellow target (desired position)
            cv2.circle(frame, (desired_x, desired_y), 8, self.COLOR_YELLOW, 2)
            
            # Draw line from current to desired
            cv2.line(frame, (current_x, current_y), (desired_x, desired_y), 
                    self.COLOR_YELLOW, 1, cv2.LINE_AA)
        
        # ================================================================
        # Draw center reference
        # ================================================================
        center_x, center_y = w // 2, h // 2
        cv2.circle(frame, (center_x, center_y), 5, self.COLOR_WHITE, 1)
        
        # ================================================================
        # Info panel - Top left
        # ================================================================
        panel_height = 200
        cv2.rectangle(overlay, (0, 0), (350, panel_height), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y_pos = 25
        line_height = 25
        
        # Title
        cv2.putText(frame, "SERVO TRACKER", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_GREEN, 2)
        y_pos += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        y_pos += line_height
        
        # Servo positions
        cv2.putText(frame, f"Pan:  {current_pan:6.1f}deg", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 1)
        y_pos += line_height
        
        cv2.putText(frame, f"Tilt: {current_tilt:6.1f}deg", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 1)
        y_pos += line_height
        
        # Detection status
        if detection is not None:
            cv2.putText(frame, f"Target: {detection.class_name}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_RED, 1)
        else:
            cv2.putText(frame, "No target", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        
        # ================================================================
        # PID values - Bottom left
        # ================================================================
        panel_y = h - 220
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, panel_y), (300, h), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        
        y_pos = panel_y + 25
        
        # Pan PID
        cv2.putText(frame, "PAN PID:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_YELLOW, 2)
        y_pos += 20
        
        cv2.putText(frame, f"P: {pan_pid.last_p_term:+7.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 18
        
        cv2.putText(frame, f"I: {pan_pid.last_i_term:+7.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 18
        
        cv2.putText(frame, f"D: {pan_pid.last_d_term:+7.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 18
        
        cv2.putText(frame, f"Out: {pan_pid.last_output:+7.2f}deg/s", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_CYAN, 1)
        y_pos += 25
        
        # Tilt PID
        cv2.putText(frame, "TILT PID:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_YELLOW, 2)
        y_pos += 20
        
        cv2.putText(frame, f"P: {tilt_pid.last_p_term:+7.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 18
        
        cv2.putText(frame, f"I: {tilt_pid.last_i_term:+7.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 18
        
        cv2.putText(frame, f"D: {tilt_pid.last_d_term:+7.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 18
        
        cv2.putText(frame, f"Out: {tilt_pid.last_output:+7.2f}deg/s", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_CYAN, 1)
        
        # ================================================================
        # Legend - Top right
        # ================================================================
        legend_x = w - 280
        legend_y = 10
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (legend_x, legend_y), (w - 10, 120), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay3, 0.6, frame, 0.4, 0, frame)
        
        y_pos = legend_y + 25
        cv2.putText(frame, "LEGEND:", (legend_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 2)
        y_pos += 22
        
        cv2.circle(frame, (legend_x + 20, y_pos - 5), 7, self.COLOR_RED, 2)
        cv2.putText(frame, "Target detected", (legend_x + 35, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 20
        
        cv2.drawMarker(frame, (legend_x + 20, y_pos - 5), self.COLOR_CYAN, 
                      cv2.MARKER_CROSS, 14, 2)
        cv2.putText(frame, "Current aim", (legend_x + 35, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        y_pos += 20
        
        cv2.circle(frame, (legend_x + 20, y_pos - 5), 6, self.COLOR_YELLOW, 2)
        cv2.putText(frame, "Desired aim", (legend_x + 35, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        
        return frame
    
    def show(self, frame: np.ndarray) -> bool:
        """
        Display frame and handle keyboard input.
        Returns False if user wants to quit.
        """
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Q or ESC to quit
        if key == ord('q') or key == 27:
            return False
        
        return True
    
    def close(self) -> None:
        """Close visualization window."""
        cv2.destroyAllWindows()


# ============================================================================
# Main Tracking System (Single Process)
# ============================================================================

class ServoTracker:
    """
    Complete servo tracking system in a single process with visualization.
    
    CONTROL LOOP:
    =============
    1. AI Camera detects object → (x, y) position in frame
    2. Convert (x, y) to desired pan/tilt angles
    3. PID calculates velocity to reach desired angles
    4. Update current angles using velocity × time
    5. Drive servos to new angles
    6. Display everything on screen
    7. Repeat at ~30 Hz
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Initialize hardware and controllers
        self.servo_driver = ServoDriver(config=config.servo)
        self.pan_pid = PIDController(gains=config.pan_pid)
        self.tilt_pid = PIDController(gains=config.tilt_pid)
        self.visualizer = TrackingVisualizer(config=config)
        
        # Current servo positions (start at center)
        self.current_pan = config.servo.pan_center
        self.current_tilt = config.servo.tilt_center
        self.desired_pan = config.servo.pan_center
        self.desired_tilt = config.servo.tilt_center
        self.running = False
        
        # For visualization
        self.latest_frame = None
        self.latest_detection = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # COCO class names (what objects the AI can detect)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow'
        ]
    
    def extract_detection(self, results: list) -> Detection | None:
        """
        Extract best detection from IMX500 results.
        
        IMX500 returns list of detections, each formatted as:
        [category, confidence, box]
        where box is [x, y, width, height] normalized 0-1
        """
        if not results:
            return None
        
        best_detection = None
        best_confidence = 0.0
        
        for detection in results:
            # IMX500 format: each detection is [category, confidence, box]
            if len(detection) < 3:
                continue
            
            category = int(detection[0])
            confidence = float(detection[1])
            box = detection[2]  # [x, y, width, height]
            
            # Filter by confidence threshold
            if confidence < self.config.confidence_threshold:
                continue
            
            # Keep highest confidence detection
            if confidence > best_confidence:
                best_confidence = confidence
                
                # Calculate center of bounding box
                center_x = box[0] + box[2] / 2.0
                center_y = box[1] + box[3] / 2.0
                
                # Map class ID to name
                class_name = self.class_names[category] if category < len(self.class_names) else 'unknown'
                
                best_detection = Detection(
                    target_x=center_x,
                    target_y=center_y,
                    confidence=confidence,
                    class_name=class_name,
                    box=list(box)
                )
        
        return best_detection
    
    def detection_to_angles(self, detection: Detection) -> tuple[float, float]:
        """Convert detection position to desired servo angles."""
        # Calculate angle offset from center
        pan_offset = (detection.target_x - 0.5) * self.config.servo.fov_horizontal
        tilt_offset = -(detection.target_y - 0.5) * self.config.servo.fov_vertical  # Invert Y
        
        # Add offset to center position
        desired_pan = self.config.servo.pan_center + pan_offset
        desired_tilt = self.config.servo.tilt_center + tilt_offset
        
        # Clamp to servo limits
        desired_pan = np.clip(desired_pan, self.config.servo.pan_min, self.config.servo.pan_max)
        desired_tilt = np.clip(desired_tilt, self.config.servo.tilt_min, self.config.servo.tilt_max)
        
        return float(desired_pan), float(desired_tilt)
    
    def update_servos(self, detection: Detection) -> None:
        """Update servo positions based on detection using PID control."""
        # Step 1: Calculate desired servo angles from detection
        self.desired_pan, self.desired_tilt = self.detection_to_angles(detection=detection)
        
        current_time = time.time()
        
        # Step 2: Calculate velocities using PID
        pan_velocity = self.pan_pid.update(
            setpoint=self.desired_pan,
            measurement=self.current_pan,
            current_time=current_time
        )
        
        tilt_velocity = self.tilt_pid.update(
            setpoint=self.desired_tilt,
            measurement=self.current_tilt,
            current_time=current_time
        )
        
        # Step 3: Update positions using velocity
        dt = 0.033  # Time delta (30 Hz update rate)
        self.current_pan += pan_velocity * dt
        self.current_tilt += tilt_velocity * dt
        
        # Step 4: Clamp to mechanical limits
        self.current_pan = float(np.clip(
            self.current_pan,
            self.config.servo.pan_min,
            self.config.servo.pan_max
        ))
        self.current_tilt = float(np.clip(
            self.current_tilt,
            self.config.servo.tilt_min,
            self.config.servo.tilt_max
        ))
        
        # Step 5: Drive servos to new positions
        self.servo_driver.set_angles(pan=self.current_pan, tilt=self.current_tilt)
    
    def run(self) -> None:
        """Main tracking loop with visualization."""
        from picamera2 import Picamera2
        from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
        
        self.running = True
        logger.info("=" * 60)
        logger.info("Servo Tracker Started (IMX500 On-Sensor AI)")
        logger.info("=" * 60)
        logger.info(f"Camera: {self.config.camera_width}x{self.config.camera_height} @ {self.config.camera_fps} FPS")
        logger.info(f"Model: {self.config.model_path}")
        logger.info(f"PID Gains - Pan: Kp={self.config.pan_pid.kp}, Ki={self.config.pan_pid.ki}, Kd={self.config.pan_pid.kd}")
        logger.info(f"PID Gains - Tilt: Kp={self.config.tilt_pid.kp}, Ki={self.config.tilt_pid.ki}, Kd={self.config.tilt_pid.kd}")
        logger.info("=" * 60)
        logger.info("Press 'Q' or 'ESC' to quit")
        
        try:
            # Initialize IMX500
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
            
            imx500 = IMX500(self.config.model_path)
            imx500.set_auto_aspect_ratio()
            
            with Picamera2(imx500.camera_num) as picam2:
                # Configure camera
                config = picam2.create_preview_configuration(
                    main={'size': (self.config.camera_width, self.config.camera_height), 'format': 'XRGB8888'},
                    controls={'FrameRate': self.config.camera_fps}
                )
                
                # Attach IMX500 for on-sensor inference
                imx500.show_network_fw_progress_bar()
                picam2.configure(config)
                
                # Set up callback to receive detections
                def callback(request):
                    # Get frame
                    self.latest_frame = request.make_array('main')
                    
                    # Get detections from metadata
                    metadata = request.get_metadata()
                    if 'ImxResults' in metadata:
                        results = metadata['ImxResults']
                        self.latest_detection = self.extract_detection(results=results)
                        
                        if self.latest_detection is not None:
                            self.update_servos(detection=self.latest_detection)
                    else:
                        self.latest_detection = None
                    
                    # Update FPS
                    self.frame_count += 1
                    if time.time() - self.last_fps_time >= 1.0:
                        self.current_fps = self.frame_count / (time.time() - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = time.time()
                
                picam2.post_callback = callback
                picam2.start()
                
                logger.info("Camera started - AI running on-sensor!")
                logger.info("Visualization window opened")
                
                # Main visualization loop
                while self.running:
                    if self.latest_frame is not None:
                        # Convert XRGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR)
                        
                        # Draw annotations
                        annotated_frame = self.visualizer.draw_annotations(
                            frame=frame_bgr,
                            detection=self.latest_detection,
                            current_pan=self.current_pan,
                            current_tilt=self.current_tilt,
                            desired_pan=self.desired_pan,
                            desired_tilt=self.desired_tilt,
                            pan_pid=self.pan_pid,
                            tilt_pid=self.tilt_pid,
                            fps=self.current_fps
                        )
                        
                        # Display
                        if not self.visualizer.show(frame=annotated_frame):
                            logger.info("User quit visualization")
                            break
                    
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Error in tracking loop: {e}", exc_info=True)
        finally:
            self.running = False
            self.servo_driver.close()
            self.visualizer.close()
            logger.info("Servo Tracker stopped")
    
    def stop(self) -> None:
        """Stop tracking."""
        self.running = False


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """
    Run the servo tracker with visualization.
    
    QUICK START:
    ============
    1. Connect AI Camera to RPi5 camera port
    2. Connect PCA9685 to I2C (GPIO 2/3)
    3. Connect servos to PCA9685 channels 0 (pan) and 1 (tilt)
    4. Power servos externally (5-6V)
    5. Run this script!
    
    VISUALIZATION:
    ==============
    - RED circle: Target detected by AI
    - CYAN crosshair: Where servos are currently pointing
    - YELLOW circle: Where servos want to go
    - Yellow line: Path from current to desired
    - PID values shown in real-time at bottom
    
    Press 'Q' or 'ESC' to quit
    
    TUNING TIPS:
    ============
    Watch the PID values in the visualization!
    
    If servos wobble/oscillate:
    - P term is too large → Reduce Kp
    - Not enough dampening → Increase Kd
    
    If servos are too slow:
    - P term is too small → Increase Kp
    
    If servos drift slightly off-target:
    - I term needed → Add small Ki
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = SystemConfig(
        servo=ServoConfig(
            # CALIBRATE THESE FOR YOUR SERVOS!
            pulse_min=150,  # Test and adjust
            pulse_max=600,  # Test and adjust
            
            # AI Camera specs (don't change unless using different camera)
            fov_horizontal=78.3,
            fov_vertical=58.7
        ),
        
        # PID tuning - start with these, adjust as needed
        pan_pid=PIDGainsConfig(
            kp=3.0,  # Proportional gain
            ki=0.2,  # Integral gain  
            kd=0.8   # Derivative gain
        ),
        tilt_pid=PIDGainsConfig(
            kp=3.0,
            ki=0.2,
            kd=0.8
        ),
        
        confidence_threshold=0.5  # Only track detections > 50% confidence
    )
    
    # Create and run tracker
    tracker = ServoTracker(config=config)
    
    try:
        tracker.run()
    except KeyboardInterrupt:
        tracker.stop()


if __name__ == "__main__":
    main()