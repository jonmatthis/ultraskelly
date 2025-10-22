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
        self.integral = np.clip(a=self.integral, a_min=-5.0, a_max=5.0)
        i_term = self.ki * self.integral
        
        # D term
        d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error
        
        # Combine and limit
        velocity = p_term + i_term + d_term
        velocity = np.clip(a=velocity, a_min=-30.0, a_max=30.0)  # Max ±30°/s
        
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
    """Controls a single servo via ServoKit."""
    
    def __init__(self, config: Config):
        self.config = config
        self.kit = None
        self._setup()
    
    def _setup(self) -> None:
        """Initialize hardware."""
        from adafruit_servokit import ServoKit
        
        self.kit = ServoKit(channels=16)
        logger.info("Servo initialized via ServoKit")
    
    def set_angle(self, angle: float) -> None:
        """Set servo angle (0-180°)."""
        angle = np.clip(a=angle, a_min=0.0, a_max=180.0)
        self.kit.servo[self.config.servo_channel].angle = angle
    
    def test_servo(self) -> None:
        """Test servo by sweeping through positions."""
        logger.info("Testing servo movement...")
        logger.info("Moving to center (90°)...")
        self.set_angle(angle=90.0)
        time.sleep(1.0)
        
        logger.info("Moving to left (60°)...")
        self.set_angle(angle=60.0)
        time.sleep(1.0)
        
        logger.info("Moving to right (120°)...")
        self.set_angle(angle=120.0)
        time.sleep(1.0)
        
        logger.info("Returning to center (90°)...")
        self.set_angle(angle=90.0)
        time.sleep(0.5)
        
        logger.info("✓ Servo test complete!")


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
        
        Args:
            frame: RGB image array
        
        Returns:
            (x, y) coordinates or None if no bright point found.
        """
        # Convert to grayscale from RGB
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            src=gray, 
            ksize=(self.config.blur_size, self.config.blur_size), 
            sigmaX=0
        )
        
        # Find the brightest point
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=blurred)
        
        # Only track if brightness is above threshold
        if max_val > 100:
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
        self.target_angle = np.clip(a=self.target_angle, a_min=0.0, a_max=180.0)
        
        # Calculate velocity using PID
        velocity = self.pid.update(target=self.target_angle, current=self.current_angle)
        
        # Update position
        dt = 0.033  # ~30 Hz
        self.current_angle += velocity * dt
        self.current_angle = float(np.clip(a=self.current_angle, a_min=0.0, a_max=180.0))
        
        # Drive servo
        self.servo.set_angle(angle=self.current_angle)
    
    def draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracking visualization.
        
        Args:
            frame: RGB image array
            
        Returns:
            RGB image with visualization overlay
        """
        h, w = frame.shape[:2]
        
        # Draw frame center line (white)
        center_x = w // 2
        cv2.line(img=frame, pt1=(center_x, 0), pt2=(center_x, h), color=(255, 255, 255), thickness=1)
        cv2.putText(
            img=frame, 
            text="CENTER", 
            org=(center_x - 40, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, 
            color=(255, 255, 255), 
            thickness=1
        )
        
        # Draw current aim point (where servo is pointing) - yellow
        angle_offset = (self.current_angle - self.config.servo_center) / self.config.camera_fov
        current_x = int((0.5 + angle_offset) * w)
        current_x = int(np.clip(a=current_x, a_min=0, a_max=w - 1))
        
        cv2.line(img=frame, pt1=(current_x, 0), pt2=(current_x, h), color=(255, 255, 0), thickness=2)
        cv2.putText(
            img=frame, 
            text="SERVO", 
            org=(current_x - 30, h - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, 
            color=(255, 255, 0), 
            thickness=2
        )
        
        # Draw brightest point - red
        if self.bright_x is not None and self.bright_y is not None:
            cv2.circle(img=frame, center=(self.bright_x, self.bright_y), radius=20, color=(255, 0, 0), thickness=3)
            cv2.putText(
                img=frame, 
                text="BRIGHTEST", 
                org=(self.bright_x - 40, self.bright_y - 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6, 
                color=(255, 0, 0), 
                thickness=2
            )
            
            # Draw line from current to target - green
            target_angle_offset = (self.target_angle - self.config.servo_center) / self.config.camera_fov
            target_x = int((0.5 + target_angle_offset) * w)
            cv2.line(img=frame, pt1=(current_x, h // 2), pt2=(target_x, h // 2), color=(0, 255, 0), thickness=2)
        
        # Info panel
        overlay = frame.copy()
        cv2.rectangle(img=overlay, pt1=(0, 0), pt2=(300, 120), color=(0, 0, 0), thickness=-1)
        cv2.addWeighted(src1=overlay, alpha=0.7, src2=frame, beta=0.3, gamma=0, dst=frame)
        
        y_pos = 25
        cv2.putText(
            img=frame, 
            text="BRIGHTNESS TRACKER", 
            org=(10, y_pos), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.6, 
            color=(0, 255, 0), 
            thickness=2
        )
        y_pos += 30
        
        cv2.putText(
            img=frame, 
            text=f"FPS: {self.fps:.1f}", 
            org=(10, y_pos),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, 
            color=(255, 255, 255), 
            thickness=1
        )
        y_pos += 25
        
        cv2.putText(
            img=frame, 
            text=f"Servo: {self.current_angle:.1f}deg", 
            org=(10, y_pos),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, 
            color=(255, 255, 0), 
            thickness=1
        )
        y_pos += 25
        
        if self.bright_x is not None:
            cv2.putText(
                img=frame, 
                text=f"Bright: ({self.bright_x}, {self.bright_y})", 
                org=(10, y_pos),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(255, 0, 0), 
                thickness=1
            )
        else:
            cv2.putText(
                img=frame, 
                text="No bright point", 
                org=(10, y_pos),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(128, 128, 128), 
                thickness=1
            )
        
        return frame
    
    def run(self) -> None:
        """Main tracking loop."""
        from picamera2 import Picamera2
        
        logger.info("=" * 50)
        logger.info("Brightness Tracker Started")
        logger.info("=" * 50)
        
        # Test servo first!
        self.servo.test_servo()
        
        logger.info(f"PID: Kp={self.config.kp}, Ki={self.config.ki}, Kd={self.config.kd}")
        logger.info("Press 'Q' to quit")
        
        try:
            picam2 = Picamera2()
            
            # Configure camera for RGB capture
            config = picam2.create_preview_configuration(
                main={
                    'size': (self.config.camera_width, self.config.camera_height),
                    'format': 'RGB888'
                }
            )
            
            picam2.configure(config)
            picam2.start()
            
            logger.info("Camera started - tracking brightest point!")
            
            # Main loop - RGB ONLY
            while True:
                # Capture frame in RGB
                frame = picam2.capture_array()
                
                # Update servo with RGB frame
                self.update(frame=frame)
                
                # Update FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # Draw visualization on RGB frame
                vis_frame = self.draw_visualization(frame=frame)
                
                # Display RGB frame
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
    2. Connect servo to ServoKit channel 0
    3. Run this script!
    
    The servo will track the brightest point in the scene.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    config = Config(
        # Servo channel (adjust to match your wiring)
        servo_channel=0,
        
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