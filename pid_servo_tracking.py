"""
Simple AI Camera Servo Tracking - No PubSub, Single Process
Direct function calls for minimum latency

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
        # The further you are from target, the harder you push.
        # Example: If error = 10° and Kp = 3.0, then P = 30°/s
        #
        # This is like a rubber band pulling you toward the target.
        # Stronger when stretched more (larger error).
        p_term = self.gains.kp * error
        
        # ================================================================
        # I TERM: Integral - "How long have I been off?"
        # ================================================================
        # Accumulate error over time. This eliminates "steady-state error"
        # (being stuck 2° off-target forever).
        #
        # Example: If you're consistently 2° too low for 5 seconds,
        # integral builds up and adds constant upward push.
        #
        # Anti-windup: Clamp integral to prevent it growing infinitely.
        # Without this, the integral can get huge and cause instability.
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10.0, 10.0)  # Anti-windup limits
        i_term = self.gains.ki * self.integral
        
        # ================================================================
        # D TERM: Derivative - "How fast am I changing?"
        # ================================================================
        # Rate of change of error. This dampens motion and prevents overshoot.
        # 
        # Example: You're approaching target fast (error decreasing rapidly).
        # D term applies brakes to slow you down before overshooting.
        #
        # Derivative = (current_error - last_error) / time_delta
        # If error is decreasing (moving toward target), derivative is negative
        # which creates a "braking" force.
        d_term = self.gains.kd * (error - self.last_error) / dt
        self.last_error = error
        
        # ================================================================
        # COMBINE ALL TERMS
        # ================================================================
        # Total output is sum of P, I, and D terms.
        # This is the velocity command (degrees per second).
        output = p_term + i_term + d_term
        
        # Clamp output to reasonable velocity limits
        # (prevents commanding impossibly fast movements)
        output = np.clip(output, -30.0, 30.0)  # Max ±30°/s
        
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
# Main Tracking System (Single Process)
# ============================================================================

class ServoTracker:
    """
    Complete servo tracking system in a single process.
    
    CONTROL LOOP:
    =============
    1. AI Camera detects object → (x, y) position in frame
    2. Convert (x, y) to desired pan/tilt angles
    3. PID calculates velocity to reach desired angles
    4. Update current angles using velocity × time
    5. Drive servos to new angles
    6. Repeat at ~30 Hz
    
    COORDINATE SYSTEMS:
    ===================
    Detection: (x, y) normalized 0.0-1.0
    - (0.5, 0.5) = center of frame
    - (0.0, 0.0) = top-left
    - (1.0, 1.0) = bottom-right
    
    Servo angles: degrees
    - 90° = center (looking straight)
    - 0° = left/down limit
    - 180° = right/up limit
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Initialize hardware and controllers
        self.servo_driver = ServoDriver(config=config.servo)
        self.pan_pid = PIDController(gains=config.pan_pid)
        self.tilt_pid = PIDController(gains=config.tilt_pid)
        
        # Current servo positions (start at center)
        self.current_pan = config.servo.pan_center
        self.current_tilt = config.servo.tilt_center
        self.running = False
        
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
        [y0, x0, y1, x1, confidence, class_id]
        
        We pick the detection with highest confidence.
        """
        if not results:
            return None
        
        best_detection = None
        best_confidence = 0.0
        
        for detection in results:
            if len(detection) < 6:
                continue
            
            y0, x0, y1, x1, confidence, class_id = detection[:6]
            
            # Filter by confidence threshold
            if confidence < self.config.confidence_threshold:
                continue
            
            # Keep highest confidence detection
            if confidence > best_confidence:
                best_confidence = confidence
                
                # Calculate center of bounding box
                center_x = (x0 + x1) / 2.0
                center_y = (y0 + y1) / 2.0
                
                # Map class ID to name
                class_name = self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else 'unknown'
                
                best_detection = Detection(
                    target_x=center_x,
                    target_y=center_y,
                    confidence=confidence,
                    class_name=class_name
                )
        
        return best_detection
    
    def detection_to_angles(self, detection: Detection) -> tuple[float, float]:
        """
        Convert detection position to desired servo angles.
        
        COORDINATE TRANSFORMATION:
        ==========================
        Detection gives us (x, y) where (0.5, 0.5) is frame center.
        We need to convert this to servo angles.
        
        Steps:
        1. Calculate offset from center: (x - 0.5)
        2. Scale by camera FOV: offset × FOV_degrees
        3. Add to servo center position
        
        Example:
        - Detection at x=0.75 (right side of frame)
        - Offset = 0.75 - 0.5 = 0.25
        - Pan offset = 0.25 × 78.3° = 19.6°
        - Desired pan = 90° + 19.6° = 109.6°
        
        Note: Y is inverted (0 = top in image, but servos increase downward)
        """
        # Calculate angle offset from center
        pan_offset = (detection.target_x - 0.5) * self.config.servo.fov_horizontal
        tilt_offset = -(detection.target_y - 0.5) * self.config.servo.fov_vertical  # Invert Y
        
        # Add offset to center position
        desired_pan = self.config.servo.pan_center + pan_offset
        desired_tilt = self.config.servo.tilt_center + tilt_offset
        
        # Clamp to servo limits (can't move beyond mechanical limits)
        desired_pan = np.clip(desired_pan, self.config.servo.pan_min, self.config.servo.pan_max)
        desired_tilt = np.clip(desired_tilt, self.config.servo.tilt_min, self.config.servo.tilt_max)
        
        return float(desired_pan), float(desired_tilt)
    
    def update_servos(self, detection: Detection) -> None:
        """
        Update servo positions based on detection using PID control.
        
        THE CONTROL LOOP:
        =================
        1. Calculate where we WANT to be (desired angles)
        2. Calculate where we ARE now (current angles)
        3. PID calculates how fast to move (velocity)
        4. Update position: new_position = old_position + velocity × time
        5. Drive servos to new position
        
        WHY USE VELOCITY CONTROL?
        =========================
        We could just set servos to desired_angle directly, but this causes:
        - Jerky motion (instant position changes)
        - Overshooting (servo momentum)
        - Oscillation (back-and-forth wobbling)
        
        Instead, PID calculates a smooth velocity that:
        - Moves fast when far from target (large P term)
        - Slows down as we approach (D term dampening)
        - Eliminates steady-state error (I term)
        """
        # Step 1: Calculate desired servo angles from detection
        desired_pan, desired_tilt = self.detection_to_angles(detection=detection)
        
        current_time = time.time()
        
        # Step 2: Calculate velocities using PID
        # PID output is in degrees/second
        pan_velocity = self.pan_pid.update(
            setpoint=desired_pan,       # Where we want to be
            measurement=self.current_pan,  # Where we are now
            current_time=current_time
        )
        
        tilt_velocity = self.tilt_pid.update(
            setpoint=desired_tilt,
            measurement=self.current_tilt,
            current_time=current_time
        )
        
        # Step 3: Update positions using velocity
        # new_position = old_position + velocity × time
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
        """
        Main tracking loop.
        
        PIPELINE:
        =========
        AI Camera (IMX500) → Detection → PID Control → Servo Driver
        
        This all runs in ONE process at ~30 FPS with minimal latency.
        """
        from picamera2 import Picamera2
        from picamera2.devices import Hailo
        
        self.running = True
        logger.info("=" * 60)
        logger.info("Servo Tracker Started")
        logger.info("=" * 60)
        logger.info(f"Camera: {self.config.camera_width}x{self.config.camera_height} @ {self.config.camera_fps} FPS")
        logger.info(f"Model: {self.config.model_path}")
        logger.info(f"PID Gains - Pan: Kp={self.config.pan_pid.kp}, Ki={self.config.pan_pid.ki}, Kd={self.config.pan_pid.kd}")
        logger.info(f"PID Gains - Tilt: Kp={self.config.tilt_pid.kp}, Ki={self.config.tilt_pid.ki}, Kd={self.config.tilt_pid.kd}")
        logger.info("=" * 60)
        
        try:
            with Picamera2() as picam2:
                # Configure camera
                main_config = {
                    'size': (self.config.camera_width, self.config.camera_height),
                    'format': 'XRGB8888'
                }
                lores_config = {
                    'size': (320, 320),  # Model input size
                    'format': 'RGB888'
                }
                controls = {'FrameRate': self.config.camera_fps}
                
                config = picam2.create_preview_configuration(
                    main=main_config,
                    lores=lores_config,
                    controls=controls
                )
                picam2.configure(config=config)
                
                # Load IMX500 AI model (runs on-sensor!)
                with Hailo(self.config.model_path) as hailo:
                    picam2.start()
                    logger.info("Camera started - AI running on-sensor (zero CPU load!)")
                    
                    frame_count = 0
                    last_log_time = time.time()
                    
                    # Main loop - runs at camera FPS (~30 Hz)
                    while self.running:
                        # Capture frame (low-res for inference)
                        frame = picam2.capture_array('lores')
                        
                        # Run AI detection (happens ON THE CAMERA SENSOR!)
                        results = hailo.run(frame)
                        
                        # Extract best detection
                        detection = self.extract_detection(results=results)
                        
                        if detection is not None:
                            # Update servos using PID control
                            # This is where the magic happens!
                            self.update_servos(detection=detection)
                        
                        # Periodic logging
                        frame_count += 1
                        if time.time() - last_log_time >= 5.0:
                            fps = frame_count / 5.0
                            logger.info(
                                f"FPS: {fps:.1f} | "
                                f"Pan: {self.current_pan:.1f}° | "
                                f"Tilt: {self.current_tilt:.1f}°"
                            )
                            frame_count = 0
                            last_log_time = time.time()
        
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        finally:
            self.running = False
            self.servo_driver.close()
            logger.info("Servo Tracker stopped")
    
    def stop(self) -> None:
        """Stop tracking."""
        self.running = False


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """
    Run the simple servo tracker.
    
    QUICK START:
    ============
    1. Connect AI Camera to RPi5 camera port
    2. Connect PCA9685 to I2C (GPIO 2/3)
    3. Connect servos to PCA9685 channels 0 (pan) and 1 (tilt)
    4. Power servos externally (5-6V)
    5. Run this script!
    
    TUNING TIPS:
    ============
    If servos wobble/oscillate:
    - Reduce Kp (make it less aggressive)
    - Increase Kd (more dampening)
    
    If servos are too slow:
    - Increase Kp (faster response)
    
    If servos drift slightly off-target:
    - Add small Ki (eliminate steady-state error)
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
