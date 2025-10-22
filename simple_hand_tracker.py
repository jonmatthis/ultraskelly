"""
Simple Brightness Tracking with One Servo
Tracks the brightest point in the scene - no AI needed!
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
    """Simple configuration for brightness tracking."""
    
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
    
    # Brightness detection
    blur_size: int = Field(default=15, description="Gaussian blur kernel size (odd number)")


# ============================================================================
# PID Controller
# ============================================================================

class PIDController:
    """Simple PID controller."""
    
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
        """Reset state."""
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
# Brightness Tracker
# ============================================================================

class BrightnessTracker:
    """Tracks the brightest point in the scene."""
    
    def __init__(self, config: Config):
        self.config = config
        self.servo = ServoDriver(config=config)
        self.pid = PIDController(kp=config.kp, ki=config.ki, kd=config.kd)
        
        # Current state
        self.current_angle = config.servo_center
        self.target_angle = config.servo_center
        self.bright_x: int | None = None
        self.bright_y: int | None = None
        
        # Visualization
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
    
    def find_brightest_point(self, frame: np.ndarray) -> tuple[int, int] | None:
        """
        Find the brightest point in the frame.
        
        Returns (x, y) coordinates or None if no bright point found.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.config.blur_size, self.config.blur_size), 0)
        
        # Find the brightest point
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        # Only track if brightness is above threshold (avoid tracking dark scenes)
        if max_val > 100:  # Adjust threshold as needed
            return max_loc  # (x, y)
        
        return None
    
    def update(self, frame: np.ndarray) -> None:
        """Update servo based on brightest point."""
        point = self.find_brightest_point(frame=frame)
        
        if point is None:
            # No bright point - reset PID
            self.pid.reset()
            self.bright_x = None
            self.bright_y = None
            return
        
        # Store for visualization
        self.bright_x, self.bright_y = point
        
        # Convert pixel position to normalized position (0-1)
        h, w = frame.shape[:2]
        norm_x = self.bright_x / w
        
        # Convert to target angle
        x_offset = (norm_x - 0.5) * self.config.camera_fov
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
        
        # Draw brightest point
        if self.bright_x is not None and self.bright_y is not None:
            cv2.circle(frame, (self.bright_x, self.bright_y), 20, (0, 0, 255), 3)
            cv2.putText(
                frame, "BRIGHTEST", (self.bright_x - 40, self.bright_y - 30),
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
        cv2.putText(frame, "BRIGHTNESS TRACKER", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25
        
        cv2.putText(frame, f"Servo: {self.current_angle:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 25
        
        if self.bright_x is not None:
            cv2.putText(frame, f"Bright: ({self.bright_x}, {self.bright_y})", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "No bright point", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
    def run(self) -> None:
        """Main tracking loop."""
        from picamera2 import Picamera2
        
        logger.info("=" * 50)
        logger.info("Brightness Tracker Started")
        logger.info("=" * 50)
        logger.info(f"PID: Kp={self.config.kp}, Ki={self.config.ki}, Kd={self.config.kd}")
        logger.info("Press 'Q' to quit")
        
        try:
            picam2 = Picamera2()
            
            # Configure camera for regular RGB capture
            config = picam2.create_preview_configuration(
                main={
                    'size': (self.config.camera_width, self.config.camera_height),
                    'format': 'RGB888'  # Request RGB format directly
                }
            )
            
            picam2.configure(config)
            picam2.start()
            
            logger.info("Camera started - tracking brightest point!")
            
            # Main loop
            while True:
                # Capture frame (already in RGB format)
                frame_rgb = picam2.capture_array()
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Update servo
                self.update(frame=frame_bgr)
                
                # Update FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # Draw visualization
                vis_frame = self.draw_visualization(frame=frame_bgr)
                
                # Display
                cv2.imshow("Brightness Tracker", vis_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        finally:
            picam2.stop()
            cv2.destroyAllWindows()
            logger.info("Brightness Tracker stopped")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """
    Run the brightness tracker!
    
    SETUP:
    1. Connect camera to RPi5
    2. Connect servo to PCA9685 channel 0
    3. Run this script!
    
    The servo will track the brightest point in the scene.
    Great for tracking flashlights, LEDs, or bright objects!
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
        kd=0.5,
        
        # Blur size (larger = smoother tracking, smaller = more responsive)
        blur_size=15
    )
    
    tracker = BrightnessTracker(config=config)
    tracker.run()


if __name__ == "__main__":
    main()