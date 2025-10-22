"""
Simple Hand Tracking with One Servo
Just centers a detected hand horizontally in the frame!
"""
import logging
import time
import numpy as np
import cv2
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config(BaseModel):
    """Simple configuration for single-servo hand tracking."""
    
    # Servo settings
    servo_channel: int = Field(default=0, description="PWM channel (0-15)")
    pulse_min: int = Field(default=150, description="PWM for 0°")
    pulse_max: int = Field(default=600, description="PWM for 180°")
    servo_center: float = Field(default=90.0, description="Center position (degrees)")
    
    # PID tuning
    kp: float = Field(default=2.0, description="Proportional gain")
    ki: float = Field(default=0.1, description="Integral gain")
    kd: float = Field(default=0.5, description="Derivative gain")
    
    # Camera
    camera_width: int = Field(default=640)
    camera_height: int = Field(default=480)
    camera_fov: float = Field(default=78.3, description="Horizontal field of view")
    
    # Hand detection model
    model_path: str = Field(
        default="/usr/share/imx500-models/imx500_network_hand_detection.rpk"
    )


# ============================================================================
# PID Controller - Smooth Tracking
# ============================================================================

class PIDController:
    """
    Simple PID controller.
    
    P: "How far am I from target?"
    I: "How long have I been off?"
    D: "How fast am I moving?"
    """
    
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time: float | None = None
    
    def update(self, target: float, current: float) -> float:
        """Calculate velocity command to reach target."""
        error = target - current
        current_time = time.time()
        
        if self.last_time is None:
            dt = 0.0
        else:
            dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt <= 0:
            return 0.0
        
        # P term
        p_term = self.kp * error
        
        # I term (with limits to prevent windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -5.0, 5.0)
        i_term = self.ki * self.integral
        
        # D term
        d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error
        
        # Combine and limit
        velocity = p_term + i_term + d_term
        velocity = np.clip(velocity, -30.0, 30.0)  # Max ±30°/s
        
        return float(velocity)
    
    def reset(self) -> None:
        """Reset state when losing target."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


# ============================================================================
# Servo Driver
# ============================================================================

class ServoDriver:
    """Controls a single servo via PCA9685."""
    
    def __init__(self, config: Config):
        self.config = config
        self.pca9685 = None
        self._setup()
    
    def _setup(self) -> None:
        """Initialize hardware."""
        try:
            from adafruit_pca9685 import PCA9685
            from board import SCL, SDA
            import busio
            
            i2c = busio.I2C(scl=SCL, sda=SDA)
            self.pca9685 = PCA9685(i2c_bus=i2c, address=0x40)
            self.pca9685.frequency = 50
            logger.info("Servo initialized")
        except Exception as e:
            logger.warning(f"No hardware: {e}")
    
    def set_angle(self, angle: float) -> None:
        """Set servo angle (0-180°)."""
        angle = np.clip(angle, 0.0, 180.0)
        pulse = int(
            self.config.pulse_min + 
            (angle / 180.0) * (self.config.pulse_max - self.config.pulse_min)
        )
        
        if self.pca9685 is not None:
            self.pca9685.channels[self.config.servo_channel].duty_cycle = pulse
        else:
            logger.debug(f"[SIM] Servo={angle:.1f}°")


# ============================================================================
# Hand Tracker
# ============================================================================

class HandTracker:
    """Single-servo hand tracking system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.servo = ServoDriver(config=config)
        self.pid = PIDController(kp=config.kp, ki=config.ki, kd=config.kd)
        
        # Current state
        self.current_angle = config.servo_center
        self.target_angle = config.servo_center
        self.hand_x: float | None = None
        
        # Visualization
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
    
    def update(self, hand_x: float | None) -> None:
        """Update servo based on detected hand position."""
        if hand_x is None:
            # No hand detected - reset PID
            self.pid.reset()
            self.hand_x = None
            return
        
        # Store for visualization
        self.hand_x = hand_x
        
        # Convert normalized x position (0-1) to target angle
        # hand_x = 0.5 should be center (servo_center degrees)
        x_offset = (hand_x - 0.5) * self.config.camera_fov
        self.target_angle = self.config.servo_center + x_offset
        self.target_angle = np.clip(self.target_angle, 0.0, 180.0)
        
        # Calculate velocity using PID
        velocity = self.pid.update(target=self.target_angle, current=self.current_angle)
        
        # Update position
        dt = 0.033  # ~30 Hz
        self.current_angle += velocity * dt
        self.current_angle = float(np.clip(self.current_angle, 0.0, 180.0))
        
        # Drive servo
        self.servo.set_angle(angle=self.current_angle)
    
    def draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking visualization."""
        h, w = frame.shape[:2]
        
        # Draw frame center line
        center_x = w // 2
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 1)
        cv2.putText(
            frame, "CENTER", (center_x - 40, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Draw current aim point (where servo is pointing)
        angle_offset = (self.current_angle - self.config.servo_center) / self.config.camera_fov
        current_x = int((0.5 + angle_offset) * w)
        current_x = np.clip(current_x, 0, w - 1)
        
        cv2.line(frame, (current_x, 0), (current_x, h), (0, 255, 255), 2)
        cv2.putText(
            frame, "SERVO", (current_x - 30, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )
        
        # Draw detected hand position
        if self.hand_x is not None:
            hand_pixel_x = int(self.hand_x * w)
            cv2.circle(frame, (hand_pixel_x, h // 2), 20, (0, 0, 255), 3)
            cv2.putText(
                frame, "HAND", (hand_pixel_x - 25, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            
            # Draw line from current to target
            target_angle_offset = (self.target_angle - self.config.servo_center) / self.config.camera_fov
            target_x = int((0.5 + target_angle_offset) * w)
            cv2.line(frame, (current_x, h // 2), (target_x, h // 2), (0, 255, 0), 2)
        
        # Info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = 25
        cv2.putText(frame, "HAND TRACKER", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        
        cv2.putText(frame, f"Servo: {self.current_angle:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 25
        
        if self.hand_x is not None:
            cv2.putText(frame, f"Hand: {self.hand_x:.2f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "No hand", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
    def run(self) -> None:
        """Main tracking loop."""
        from picamera2 import Picamera2
        from picamera2.devices.imx500 import IMX500
        
        logger.info("=" * 50)
        logger.info("Hand Tracker Started")
        logger.info("=" * 50)
        logger.info(f"PID: Kp={self.config.kp}, Ki={self.config.ki}, Kd={self.config.kd}")
        logger.info("Press 'Q' to quit")
        
        try:
            # Set up camera with hand detection
            imx500 = IMX500(self.config.model_path)
            imx500.set_auto_aspect_ratio()
            
            with Picamera2(imx500.camera_num) as picam2:
                config = picam2.create_preview_configuration(
                    main={
                        'size': (self.config.camera_width, self.config.camera_height),
                        'format': 'XRGB8888'
                    },
                    controls={'FrameRate': 30}
                )
                
                imx500.show_network_fw_progress_bar()
                picam2.configure(config)
                
                latest_frame = None
                latest_hand_x = None
                
                def callback(request) -> None:
                    nonlocal latest_frame, latest_hand_x
                    
                    # Get frame
                    latest_frame = request.make_array('main')
                    
                    # Get hand detection
                    metadata = request.get_metadata()
                    latest_hand_x = None
                    
                    if 'ImxResults' in metadata:
                        results = metadata['ImxResults']
                        if results and len(results) > 0:
                            # Get first hand detection
                            detection = results[0]
                            if len(detection) >= 3:
                                box = detection[2]  # [x, y, width, height]
                                # Calculate center x
                                latest_hand_x = float(box[0] + box[2] / 2.0)
                    
                    # Update servo
                    self.update(hand_x=latest_hand_x)
                    
                    # Update FPS
                    self.frame_count += 1
                    if time.time() - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count / (time.time() - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = time.time()
                
                picam2.post_callback = callback
                picam2.start()
                
                logger.info("Camera started - looking for hands!")
                
                # Visualization loop
                while True:
                    if latest_frame is not None:
                        # Convert to BGR
                        frame_bgr = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
                        
                        # Draw visualization
                        vis_frame = self.draw_visualization(frame=frame_bgr)
                        
                        # Display
                        cv2.imshow("Hand Tracker", vis_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        finally:
            cv2.destroyAllWindows()
            logger.info("Hand Tracker stopped")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """
    Run the hand tracker!
    
    SETUP:
    1. Connect AI Camera to RPi5
    2. Connect servo to PCA9685 channel 0
    3. Make sure hand detection model is installed
    4. Run this script!
    
    The servo will try to keep any detected hand centered in the frame.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    config = Config(
        # Tune these for your servo
        pulse_min=150,
        pulse_max=600,
        
        # PID tuning (adjust if too fast/slow/wobbly)
        kp=2.0,
        ki=0.1,
        kd=0.5
    )
    
    tracker = HandTracker(config=config)
    tracker.run()


if __name__ == "__main__":
    main()
