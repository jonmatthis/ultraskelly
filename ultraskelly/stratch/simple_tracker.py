"""
Dead Simple 2-Axis Brightness Tracker
Pan + Tilt - tracks bright point in X and Y
"""
import logging
logger = logging.getLogger(__name__)
import  time
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
PROPORTIONAL_GAIN: float = 0.05  # How aggressively to move
DEADZONE_PIXELS: int = 30  # Don't move servo if error is within this range

# Servo limits
PAN_MIN: float = 0.0
PAN_MAX: float = 180.0
TILT_MIN: float = 0.0
TILT_MAX: float = 180.0


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
    
    # Deadzone visualization (gray box)
    deadzone_left = center_x - DEADZONE_PIXELS
    deadzone_right = center_x + DEADZONE_PIXELS
    deadzone_top = center_y - DEADZONE_PIXELS
    deadzone_bottom = center_y + DEADZONE_PIXELS
    cv2.rectangle(img=frame, pt1=(deadzone_left, deadzone_top), 
                  pt2=(deadzone_right, deadzone_bottom), 
                  color=(128, 128, 128), thickness=1)
    
    # Brightest point
    if bright_x is not None and bright_y is not None:
        color = (0, 255, 0) if (is_locked_x and is_locked_y) else (255, 0, 0)
        cv2.circle(img=frame, center=(bright_x, bright_y), radius=20, color=color, thickness=3)
        
        # Draw lines from center to bright point
        line_color_x = (0, 255, 0) if is_locked_x else (255, 255, 0)
        line_color_y = (0, 255, 0) if is_locked_y else (255, 255, 0)
        cv2.line(img=frame, pt1=(center_x, bright_y), pt2=(bright_x, bright_y), 
                 color=line_color_x, thickness=2)
        cv2.line(img=frame, pt1=(bright_x, center_y), pt2=(bright_x, bright_y), 
                 color=line_color_y, thickness=2)
    
    # Info overlay
    status = "LOCKED" if (is_locked_x and is_locked_y) else "TRACKING"
    status_color = (0, 255, 0) if (is_locked_x and is_locked_y) else (255, 255, 0)
    cv2.putText(img=frame, text=status, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
                color=status_color, thickness=2)
    
    cv2.putText(img=frame, text=f"Pan: {pan_angle:.1f}° {'✓' if is_locked_x else ''}", 
                org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                color=(255, 255, 255), thickness=1)
    
    cv2.putText(img=frame, text=f"Tilt: {tilt_angle:.1f}° {'✓' if is_locked_y else ''}", 
                org=(10, 85), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                color=(255, 255, 255), thickness=1)
    
    cv2.putText(img=frame, text=f"FPS: {fps:.1f}", org=(10, 110),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                color=(255, 255, 255), thickness=1)
    
    if bright_x is not None and bright_y is not None:
        error_x = bright_x - center_x
        error_y = bright_y - center_y
        cv2.putText(img=frame, text=f"Error: X={error_x:+4d}px Y={error_y:+4d}px", 
                    org=(10, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, 
                    color=(255, 255, 255), thickness=1)
    
    return frame


def main() -> None:
    """Run the 2-axis tracker."""
    logger.info("="*60)
    logger.info("2-AXIS BRIGHTNESS TRACKER")
    logger.info("="*60)
    
    # Initialize servos
    kit = ServoKit(channels=16)
    pan_angle = 90.0
    tilt_angle = 90.0
    kit.servo[PAN_SERVO_CHANNEL].angle = pan_angle
    kit.servo[TILT_SERVO_CHANNEL].angle = tilt_angle
    time.sleep(0.5)
    
    # Test servos
    logger.info("Testing pan servo...")
    kit.servo[PAN_SERVO_CHANNEL].angle = 60.0
    time.sleep(0.5)
    kit.servo[PAN_SERVO_CHANNEL].angle = 120.0
    time.sleep(0.5)
    kit.servo[PAN_SERVO_CHANNEL].angle = 90.0
    time.sleep(0.5)
    
    logger.info("Testing tilt servo...")
    kit.servo[TILT_SERVO_CHANNEL].angle = 60.0
    time.sleep(0.5)
    kit.servo[TILT_SERVO_CHANNEL].angle = 120.0
    time.sleep(0.5)
    kit.servo[TILT_SERVO_CHANNEL].angle = 90.0
    time.sleep(0.5)
    
    logger.info("Servo tests complete!")
    
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
                
                # Calculate errors
                center_x = CAMERA_WIDTH // 2
                center_y = CAMERA_HEIGHT // 2
                error_x = bright_x - center_x
                error_y = bright_y - center_y
                
                # Check if locked
                is_locked_x = abs(error_x) <= DEADZONE_PIXELS
                is_locked_y = abs(error_y) <= DEADZONE_PIXELS
                
                # Pan control (X axis)
                if not is_locked_x:
                    pan_angle += error_x * PROPORTIONAL_GAIN
                    pan_angle = np.clip(a=pan_angle, a_min=PAN_MIN, a_max=PAN_MAX)
                    kit.servo[PAN_SERVO_CHANNEL].angle = pan_angle
                
                # Tilt control (Y axis)
                if not is_locked_y:
                    tilt_angle += error_y * PROPORTIONAL_GAIN
                    tilt_angle = np.clip(a=tilt_angle, a_min=TILT_MIN, a_max=TILT_MAX)
                    kit.servo[TILT_SERVO_CHANNEL].angle = tilt_angle
                
                logger.debug(f"X: {error_x:+4d}px {'✓' if is_locked_x else ' '} | "
                           f"Y: {error_y:+4d}px {'✓' if is_locked_y else ' '} | "
                           f"Pan: {pan_angle:.1f}° | Tilt: {tilt_angle:.1f}°")
            else:
                bright_x, bright_y = None, None
                is_locked_x = False
                is_locked_y = False
            
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
                pan_angle=pan_angle,
                tilt_angle=tilt_angle,
                fps=fps,
                is_locked_x=is_locked_x,
                is_locked_y=is_locked_y
            )
            cv2.imshow("2-Axis Brightness Tracker", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        # Center and release servos
        logger.info("Releasing servos...")
        kit.servo[PAN_SERVO_CHANNEL].angle = 90.0
        kit.servo[TILT_SERVO_CHANNEL].angle = 90.0
        time.sleep(0.5)
        kit.servo[PAN_SERVO_CHANNEL].angle = None
        kit.servo[TILT_SERVO_CHANNEL].angle = None
        
        picam2.stop()
        cv2.destroyAllWindows()
        logger.info("Done!")


if __name__ == "__main__":
    main()