"""
Face Detection 2-Axis Tracker using Hailo AI Accelerator
Pan + Tilt - tracks detected face center in X and Y
"""
import logging
import time
import numpy as np
import cv2
from adafruit_servokit import ServoKit
from picamera2 import Picamera2
from picamera2.devices import Hailo

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
PAN_SERVO_CHANNEL: int = 0   # Horizontal movement
TILT_SERVO_CHANNEL: int = 7  # Vertical movement

CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 960

# Hailo Model Configuration
# Options: scrfd_2.5g, scrfd_10g, scrfd_500m, retinaface_mobilenet_v1
FACE_MODEL: str = "scrfd_2.5g"
MODEL_PATH: str = f"/usr/share/hailo-models/{FACE_MODEL}_h8l.hef"
CONFIDENCE_THRESHOLD: float = 0.5

# Control tuning
PROPORTIONAL_GAIN: float = 0.05  # How aggressively to move
DEADZONE_PIXELS: int = 40  # Don't move servo if error is within this range

# Servo limits
PAN_MIN: float = 0.0
PAN_MAX: float = 180.0
TILT_MIN: float = 0.0
TILT_MAX: float = 180.0


# ============================================================================
# Face Detection and Tracking
# ============================================================================

def extract_face_detections(
    results: list, 
    confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> list[dict[str, float | tuple[float, float]]]:
    """Extract face detections from Hailo results.
    
    Returns list of face dicts with 'bbox', 'confidence', 'center', 'landmarks'
    """
    faces: list[dict[str, float | tuple[float, float]]] = []
    
    if not results or len(results) == 0:
        return faces
    
    # Hailo returns detections with bboxes and landmarks
    detections = results[0]  # First output contains detection results
    
    for detection in detections:
        # Extract confidence score
        confidence = float(detection.get('confidence', 0.0))
        
        if confidence < confidence_threshold:
            continue
        
        # Extract bounding box [x_min, y_min, x_max, y_max]
        bbox = detection.get('bbox', [])
        if len(bbox) < 4:
            continue
            
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate center
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        
        # Extract landmarks if available (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        landmarks = detection.get('landmarks', [])
        
        faces.append({
            'bbox': (float(x_min), float(y_min), float(x_max), float(y_max)),
            'confidence': confidence,
            'center': (center_x, center_y),
            'landmarks': landmarks
        })
    
    return faces


def select_primary_face(
    faces: list[dict[str, float | tuple[float, float]]], 
    prev_center: tuple[float, float] | None = None
) -> dict[str, float | tuple[float, float]] | None:
    """Select the primary face to track.
    
    Priority:
    1. If prev_center exists, select closest face (temporal consistency)
    2. Otherwise, select largest face (by bbox area)
    """
    if not faces:
        return None
    
    if prev_center is not None:
        # Select face closest to previous position
        prev_x, prev_y = prev_center
        min_dist = float('inf')
        closest_face = None
        
        for face in faces:
            face_x, face_y = face['center']
            dist = np.sqrt((face_x - prev_x)**2 + (face_y - prev_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_face = face
        
        return closest_face
    else:
        # Select largest face
        max_area = 0.0
        largest_face = None
        
        for face in faces:
            x_min, y_min, x_max, y_max = face['bbox']
            area = (x_max - x_min) * (y_max - y_min)
            if area > max_area:
                max_area = area
                largest_face = face
        
        return largest_face


def draw_visualization(
    frame: np.ndarray,
    face: dict[str, float | tuple[float, float]] | None,
    pan_angle: float,
    tilt_angle: float,
    fps: float,
    is_locked_x: bool = False,
    is_locked_y: bool = False,
    num_faces: int = 0
) -> np.ndarray:
    """Draw visualization overlay."""
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
    cv2.rectangle(
        img=frame,
        pt1=(deadzone_left, deadzone_top),
        pt2=(deadzone_right, deadzone_bottom),
        color=(128, 128, 128),
        thickness=1
    )
    
    # Draw primary face
    if face is not None:
        # Face bounding box
        x_min, y_min, x_max, y_max = face['bbox']
        x_min_int = int(x_min)
        y_min_int = int(y_min)
        x_max_int = int(x_max)
        y_max_int = int(y_max)
        
        box_color = (0, 255, 0) if (is_locked_x and is_locked_y) else (255, 165, 0)
        cv2.rectangle(
            img=frame,
            pt1=(x_min_int, y_min_int),
            pt2=(x_max_int, y_max_int),
            color=box_color,
            thickness=2
        )
        
        # Face center
        face_x, face_y = face['center']
        face_x_int = int(face_x)
        face_y_int = int(face_y)
        cv2.circle(
            img=frame,
            center=(face_x_int, face_y_int),
            radius=8,
            color=box_color,
            thickness=-1
        )
        
        # Draw landmarks if available
        landmarks = face.get('landmarks', [])
        if landmarks and len(landmarks) >= 10:  # 5 points x 2 coordinates
            for i in range(0, len(landmarks), 2):
                if i + 1 < len(landmarks):
                    lm_x = int(landmarks[i])
                    lm_y = int(landmarks[i + 1])
                    cv2.circle(img=frame, center=(lm_x, lm_y), radius=3, color=(0, 255, 255), thickness=-1)
        
        # Lines from center to face
        line_color_x = (0, 255, 0) if is_locked_x else (255, 255, 0)
        line_color_y = (0, 255, 0) if is_locked_y else (255, 255, 0)
        cv2.line(
            img=frame,
            pt1=(center_x, face_y_int),
            pt2=(face_x_int, face_y_int),
            color=line_color_x,
            thickness=2
        )
        cv2.line(
            img=frame,
            pt1=(face_x_int, center_y),
            pt2=(face_x_int, face_y_int),
            color=line_color_y,
            thickness=2
        )
        
        # Confidence
        confidence = face['confidence']
        cv2.putText(
            img=frame,
            text=f"Conf: {confidence:.2f}",
            org=(x_min_int, y_min_int - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=box_color,
            thickness=2
        )
    
    # Status overlay
    status = "LOCKED" if (is_locked_x and is_locked_y and face) else ("TRACKING" if face else "SEARCHING")
    status_color = (0, 255, 0) if status == "LOCKED" else ((255, 165, 0) if status == "TRACKING" else (255, 0, 0))
    cv2.putText(
        img=frame,
        text=status,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=status_color,
        thickness=2
    )
    
    cv2.putText(
        img=frame,
        text=f"Faces: {num_faces}",
        org=(10, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=1
    )
    
    cv2.putText(
        img=frame,
        text=f"Pan: {pan_angle:.1f}° {'✓' if is_locked_x else ''}",
        org=(10, 90),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=1
    )
    
    cv2.putText(
        img=frame,
        text=f"Tilt: {tilt_angle:.1f}° {'✓' if is_locked_y else ''}",
        org=(10, 115),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=1
    )
    
    cv2.putText(
        img=frame,
        text=f"FPS: {fps:.1f}",
        org=(10, 140),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(255, 255, 255),
        thickness=1
    )
    
    if face:
        face_x, face_y = face['center']
        error_x = int(face_x - center_x)
        error_y = int(face_y - center_y)
        cv2.putText(
            img=frame,
            text=f"Error: X={error_x:+4d}px Y={error_y:+4d}px",
            org=(10, 165),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=1
        )
    
    return frame


def main() -> None:
    """Run the face tracking system."""
    logger.info("="*60)
    logger.info("HAILO AI FACE TRACKER")
    logger.info(f"Model: {FACE_MODEL}")
    logger.info("="*60)
    
    # Initialize servos
    kit = ServoKit(channels=16)
    pan_angle = 90.0
    tilt_angle = 90.0
    kit.servo[PAN_SERVO_CHANNEL].angle = pan_angle
    kit.servo[TILT_SERVO_CHANNEL].angle = tilt_angle
    time.sleep(delay=0.5)
    
    # Test servos
    logger.info("Testing servos...")
    for test_angle in [60.0, 120.0, 90.0]:
        kit.servo[PAN_SERVO_CHANNEL].angle = test_angle
        time.sleep(delay=0.3)
    for test_angle in [60.0, 120.0, 90.0]:
        kit.servo[TILT_SERVO_CHANNEL].angle = test_angle
        time.sleep(delay=0.3)
    logger.info("Servo tests complete!")
    
    # Initialize Hailo and camera
    logger.info("Initializing Hailo AI accelerator...")
    try:
        with Hailo(MODEL_PATH) as hailo:
            model_h, model_w, _ = hailo.get_input_shape()
            logger.info(f"Model input size: {model_w}x{model_h}")
            
            # Initialize camera
            logger.info("Starting camera...")
            picam2 = Picamera2()
            
            # Dual stream: main for display, lores for inference
            main = {'size': (CAMERA_WIDTH, CAMERA_HEIGHT), 'format': 'RGB888'}
            lores = {'size': (model_w, model_h), 'format': 'RGB888'}
            config = picam2.create_preview_configuration(main=main, lores=lores)
            picam2.configure(config=config)
            picam2.start()
            time.sleep(delay=1)
            
            logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
            logger.info(f"Proportional gain: {PROPORTIONAL_GAIN}")
            logger.info(f"Deadzone: ±{DEADZONE_PIXELS}px")
            logger.info("Press 'Q' to quit")
            logger.info("="*60)
            
            # FPS tracking
            frame_count = 0
            last_fps_time = time.time()
            fps = 0.0
            
            # Track previous face position
            prev_face_center: tuple[float, float] | None = None
            
            try:
                while True:
                    # Capture frames
                    main_frame = picam2.capture_array(name='main')
                    lores_frame = picam2.capture_array(name='lores')
                    
                    # Run face detection on low-res frame
                    results = hailo.run(lores_frame)
                    
                    # Extract face detections
                    faces = extract_face_detections(results=results, confidence_threshold=CONFIDENCE_THRESHOLD)
                    
                    # Select primary face to track
                    primary_face = select_primary_face(faces=faces, prev_center=prev_face_center)
                    
                    is_locked_x = False
                    is_locked_y = False
                    
                    if primary_face is not None:
                        # Scale face coordinates from model resolution to camera resolution
                        scale_x = CAMERA_WIDTH / model_w
                        scale_y = CAMERA_HEIGHT / model_h
                        
                        face_x, face_y = primary_face['center']
                        face_x *= scale_x
                        face_y *= scale_y
                        
                        # Update tracking
                        prev_face_center = (face_x, face_y)
                        
                        # Calculate errors
                        center_x = CAMERA_WIDTH // 2
                        center_y = CAMERA_HEIGHT // 2
                        error_x = face_x - center_x
                        error_y = face_y - center_y
                        
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
                        
                        # Scale bbox for display
                        x_min, y_min, x_max, y_max = primary_face['bbox']
                        primary_face['bbox'] = (
                            x_min * scale_x,
                            y_min * scale_y,
                            x_max * scale_x,
                            y_max * scale_y
                        )
                        primary_face['center'] = (face_x, face_y)
                        
                        # Scale landmarks
                        if 'landmarks' in primary_face and primary_face['landmarks']:
                            scaled_landmarks = []
                            for i, val in enumerate(primary_face['landmarks']):
                                if i % 2 == 0:  # x coordinate
                                    scaled_landmarks.append(val * scale_x)
                                else:  # y coordinate
                                    scaled_landmarks.append(val * scale_y)
                            primary_face['landmarks'] = scaled_landmarks
                    else:
                        prev_face_center = None
                    
                    # Update FPS
                    frame_count += 1
                    if time.time() - last_fps_time >= 1.0:
                        fps = frame_count / (time.time() - last_fps_time)
                        frame_count = 0
                        last_fps_time = time.time()
                    
                    # Visualize
                    vis_frame = draw_visualization(
                        frame=main_frame,
                        face=primary_face,
                        pan_angle=pan_angle,
                        tilt_angle=tilt_angle,
                        fps=fps,
                        is_locked_x=is_locked_x,
                        is_locked_y=is_locked_y,
                        num_faces=len(faces)
                    )
                    cv2.imshow(winname="Hailo Face Tracker", mat=vis_frame)
                    
                    if cv2.waitKey(delay=1) & 0xFF == ord('q'):
                        break
            
            except KeyboardInterrupt:
                logger.info("\nStopped by user")
            finally:
                # Center and release servos
                logger.info("Releasing servos...")
                kit.servo[PAN_SERVO_CHANNEL].angle = 90.0
                kit.servo[TILT_SERVO_CHANNEL].angle = 90.0
                time.sleep(delay=0.5)
                kit.servo[PAN_SERVO_CHANNEL].angle = None
                kit.servo[TILT_SERVO_CHANNEL].angle = None
                
                picam2.stop()
                cv2.destroyAllWindows()
                logger.info("Done!")
    
    except Exception as e:
        logger.error(f"Error initializing Hailo: {e}")
        logger.error(f"Make sure the model file exists at: {MODEL_PATH}")
        logger.error("Available models: scrfd_2.5g, scrfd_10g, scrfd_500m, retinaface_mobilenet_v1")
        raise


if __name__ == "__main__":
    main()
