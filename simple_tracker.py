def main() -> None:
    """Run the simple tracker."""
    logger.info("="*60)
    logger.info("2-AXIS BRIGHTNESS TRACKER")
    logger.info("="*60)
    
    # Initialize servos
    kit = ServoKit(channels=16)
    pan_angle = 90.0
    tilt_angle = 90.0
    kit.servo[PAN_SERVO_CHANNEL].angle = pan_angle
    kit.servo[TILT_SERVO_CHANNEL].angle = tilt_angle
    
    # Test servos
    logger.info("Testing servos...")
    logger.info("Pan left...")
    kit.servo[PAN_SERVO_CHANNEL].angle = 60.0
    time.sleep(0.5)
    logger.info("Pan right...")
    kit.servo[PAN_SERVO_CHANNEL].angle = 120.0
    time.sleep(0.5)
    logger.info("Pan center...")
    kit.servo[PAN_SERVO_CHANNEL].angle = 90.0
    time.sleep(0.5)
    
    logger.info("Tilt up...")
    kit.servo["""
Dead Simple Brightness Tracker
Just proportional control - bright point off center? Move servo towards it. Done.
"""
import logging
import time
import numpy as np
import cv2
from adafruit_servokit import ServoKit
from picamera2 import Picamera2

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
PAN_SERVO_CHANNEL: int = 0   # Horizontal movement
TILT_SERVO_CHANNEL: int = 7  # Vertical movement

CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
BLUR_SIZE: int = 15

# Control tuning
PROPORTIONAL_GAIN: float = 0.05 # How aggressively to move (0.1 = gentle, 0.5 = aggressive)
DEADZONE_PIXELS: int = 30  # Don't move servo if error is within this range
PAN_MIN: float = 30.0
PAN_MAX: float = 150.0
TILT_MIN: float = 30.0
TILT_MAX: float = 150.0


# ============================================================================
# Main Tracker
# ============================================================================

def find_brightest_point(frame: np.ndarray) -> tuple[int, int] | None:
    """Find the brightest point in RGB frame."""
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(BLUR_SIZE, BLUR_SIZE), sigmaX=0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=blurred)
    
    if max_val > 100:
        return max_loc  # (x, y)
    return None


def draw_visualization(frame: np.ndarray, bright_x: int | None, bright_y: int | None, 
                       pan_angle: float, tilt_angle: float, fps: float, 
                       is_locked_x: bool = False, is_locked_y: bool = False) -> np.ndarray:
    """Draw simple visualization."""
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2
    
    # Center crosshair (white)
    cv2.line(img=frame, pt1=(center_x, 0), pt2=(center_x, h), color=(255, 255, 255), thickness=2)
    cv2.line(img=frame, pt1=(0, center_y), pt2=(w, center_y), color=(255, 255, 255), thickness=2)
    
    # Deadzone visualization (light gray box)
    deadzone_left = center_x - DEADZONE_PIXELS
    deadzone_right = center_x + DEADZONE_PIXELS
    deadzone_top = center_y - DEADZONE_PIXELS
    deadzone_bottom = center_y + DEADZONE_PIXELS
    cv2.rectangle(img=frame, pt1=(deadzone_left, deadzone_top), pt2=(deadzone_right, deadzone_bottom), 
                  color=(128, 128, 128), thickness=1)
    
    # Brightest point (red circle, green if both axes locked)
    if bright_x is not None and bright_y is not None:
        color = (0, 255, 0) if (is_locked_x and is_locked_y) else (255, 0, 0)
        cv2.circle(img=frame, center=(bright_x, bright_y), radius=20, color=color, thickness=3)
        
        # Draw crosshair from center to bright point
        line_color_x = (0, 255, 0) if is_locked_x else (255, 255, 0)
        line_color_y = (0, 255, 0) if is_locked_y else (255, 255, 0)
        cv2.line(img=frame, pt1=(center_x, bright_y), pt2=(bright_x, bright_y), color=line_color_x, thickness=2)
        cv2.line(img=frame, pt1=(bright_x, center_y), pt2=(bright_x, bright_y), color=line_color_y, thickness=2)
    
    # Info
    status = "LOCKED" if (is_locked_x and is_locked_y) else "TRACKING"
    status_color = (0, 255, 0) if (is_locked_x and is_locked_y) else (255, 255, 0)
    cv2.putText(img=frame, text=status, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=status_color, thickness=2)
    
    cv2.putText(img=frame, text=f"Pan: {pan_angle:.1f}deg {'✓' if is_locked_x else ''}", org=(10, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    cv2.putText(img=frame, text=f"Tilt: {tilt_angle:.1f}deg {'✓' if is_locked_y else ''}", org=(10, 85),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    cv2.putText(img=frame, text=f"FPS: {fps:.1f}", org=(10, 110),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    
    if bright_x is not None and bright_y is not None:
        error_x = bright_x - center_x
        error_y = bright_y - center_y
        cv2.putText(img=frame, text=f"Error: X={error_x:+4d}px Y={error_y:+4d}px", org=(10, 135),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    
    return frame


def main() -> None:
    """Run the simple tracker."""
    logger.info("="*60)
    logger.info("SIMPLE BRIGHTNESS TRACKER")
    logger.info("="*60)
    
    # Initialize servo
    kit = ServoKit(channels=16)
    servo_angle = 90.0
    kit.servo[SERVO_CHANNEL].angle = servo_angle
    
    # Test servo
    logger.info("Testing servo...")
    kit.servo[SERVO_CHANNEL].angle = 60.0
    time.sleep(0.5)
    kit.servo[SERVO_CHANNEL].angle = 120.0
    time.sleep(0.5)
    kit.servo[SERVO_CHANNEL].angle = 90.0
    time.sleep(0.5)
    logger.info("Servo test complete!")
    
    # Initialize camera
    logger.info("Starting camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={'size': (CAMERA_WIDTH, CAMERA_HEIGHT), 'format': 'RGB888'}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    
    logger.info(f"Proportional gain: {PROPORTIONAL_GAIN}")
    logger.info(f"Deadzone: ±{DEADZONE_PIXELS}px")
    logger.info("Press 'Q' to quit")
    logger.info("="*60)
    
    # FPS tracking
    frame_count = 0
    last_fps_time = time.time()
    fps = 0.0
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Find brightest point
            point = find_brightest_point(frame=frame)
            
            if point is not None:
                bright_x, bright_y = point
                
                # Calculate error (how far from center)
                center_x = CAMERA_WIDTH // 2
                error_pixels = bright_x - center_x
                
                # Only move if outside deadzone
                is_locked = abs(error_pixels) <= DEADZONE_PIXELS
                
                if not is_locked:
                    # Proportional control: move servo proportionally to error
                    servo_angle += error_pixels * PROPORTIONAL_GAIN
                    servo_angle = np.clip(a=servo_angle, a_min=SERVO_MIN, a_max=SERVO_MAX)
                    
                    # Set servo
                    kit.servo[SERVO_CHANNEL].angle = servo_angle
                    
                    logger.debug(f"Error: {error_pixels:4d}px | Servo: {servo_angle:6.1f}°")
                else:
                    logger.debug(f"Error: {error_pixels:4d}px | LOCKED (in deadzone)")
            else:
                bright_x, bright_y = None, None
                is_locked = False
            
            # Update FPS
            frame_count += 1
            if time.time() - last_fps_time >= 1.0:
                fps = frame_count / (time.time() - last_fps_time)
                frame_count = 0
                last_fps_time = time.time()
            
            # Visualize
            vis_frame = draw_visualization(
                frame=frame, 
                bright_x=bright_x, 
                bright_y=bright_y,
                servo_angle=servo_angle,
                fps=fps,
                is_locked=is_locked
            )
            cv2.imshow("Brightness Tracker", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        # Center servo then release it
        logger.info("Releasing servo...")
        kit.servo[SERVO_CHANNEL].angle = 90.0
        time.sleep(0.5)
        kit.servo[SERVO_CHANNEL].angle = None  # Release the servo (de-energize)
        
        picam2.stop()
        cv2.destroyAllWindows()
        logger.info("Done!")


if __name__ == "__main__":
    main()
