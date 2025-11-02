"""
Face Detection 2-Axis Tracker using Hailo AI Accelerator
Properly handles SCRFD model output format
"""
import logging
import time
from typing import Optional
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
PAN_SERVO_CHANNEL: int = 0
TILT_SERVO_CHANNEL: int = 7

CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 960

# Model configuration
FACE_MODEL: str = "scrfd_2.5g"
MODEL_PATH: str = f"/usr/share/hailo-models/{FACE_MODEL}_h8l.hef"
CONFIDENCE_THRESHOLD: float = 0.5

# Control parameters
PROPORTIONAL_GAIN: float = 0.05
DEADZONE_PIXELS: int = 40

# Servo limits
PAN_MIN: float = 0.0
PAN_MAX: float = 180.0
TILT_MIN: float = 0.0
TILT_MAX: float = 180.0


# ============================================================================
# SCRFD Post-Processing
# ============================================================================

class SCRFDPostProcessor:
    """Post-process raw SCRFD model outputs from Hailo."""
    
    def __init__(
        self,
        model_input_size: tuple[int, int],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> None:
        self.input_height: int = model_input_size[0]
        self.input_width: int = model_input_size[1]
        self.confidence_threshold: float = confidence_threshold
        self.nms_threshold: float = nms_threshold
        
        # SCRFD anchor configuration for 640x640 input
        self.steps: list[int] = [8, 16, 32]
        self.min_sizes: list[list[int]] = [[16, 32], [64, 128], [256, 512]]
        self.anchors: np.ndarray = self._generate_anchors()
    
    def _generate_anchors(self) -> np.ndarray:
        """Generate anchor boxes for all feature map levels."""
        all_anchors: list[np.ndarray] = []
        
        for stride, min_size_list in zip(self.steps, self.min_sizes):
            h: int = self.input_height // stride
            w: int = self.input_width // stride
            num_anchors: int = len(min_size_list)
            
            # Create grid centers
            y_centers: np.ndarray
            x_centers: np.ndarray
            y_centers, x_centers = np.mgrid[0:h, 0:w]
            centers: np.ndarray = np.stack([x_centers, y_centers], axis=-1).astype(dtype=np.float32)
            centers = (centers * stride).reshape((-1, 2))
            
            # Normalize to [0, 1]
            centers[:, 0] /= self.input_width
            centers[:, 1] /= self.input_height
            
            # Replicate for multiple anchors per location
            if num_anchors > 1:
                centers = np.repeat(centers[np.newaxis, :, :], repeats=num_anchors, axis=0)
                centers = centers.reshape((-1, 2))
            
            # Create scales
            scales: np.ndarray = np.ones_like(centers) * stride
            scales[:, 0] /= self.input_width
            scales[:, 1] /= self.input_height
            
            # Concatenate [center_x, center_y, scale_x, scale_y]
            anchors: np.ndarray = np.concatenate([centers, scales], axis=1)
            all_anchors.append(anchors)
        
        return np.concatenate(all_anchors, axis=0)
    
    def _decode_boxes(self, box_preds: np.ndarray) -> np.ndarray:
        """Decode bounding box predictions using anchors."""
        # box_preds: [N, 4] where 4 = [dx1, dy1, dx2, dy2]
        x1: np.ndarray = self.anchors[:, 0] - box_preds[:, 0] * self.anchors[:, 2]
        y1: np.ndarray = self.anchors[:, 1] - box_preds[:, 1] * self.anchors[:, 3]
        x2: np.ndarray = self.anchors[:, 0] + box_preds[:, 2] * self.anchors[:, 2]
        y2: np.ndarray = self.anchors[:, 1] + box_preds[:, 3] * self.anchors[:, 3]
        
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def _decode_landmarks(self, landmark_preds: np.ndarray) -> np.ndarray:
        """Decode facial landmark predictions using anchors."""
        # landmark_preds: [N, 10] where 10 = 5 keypoints * 2 coords
        landmarks: list[np.ndarray] = []
        for i in range(0, 10, 2):
            x: np.ndarray = self.anchors[:, 0] + landmark_preds[:, i] * self.anchors[:, 2]
            y: np.ndarray = self.anchors[:, 1] + landmark_preds[:, i + 1] * self.anchors[:, 3]
            landmarks.extend([x, y])
        
        return np.stack(landmarks, axis=-1)
    
    def _nms(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        landmarks: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply Non-Maximum Suppression."""
        if len(boxes) == 0:
            return boxes, scores, landmarks
        
        x1: np.ndarray = boxes[:, 0]
        y1: np.ndarray = boxes[:, 1]
        x2: np.ndarray = boxes[:, 2]
        y2: np.ndarray = boxes[:, 3]
        
        areas: np.ndarray = (x2 - x1) * (y2 - y1)
        order: np.ndarray = scores.argsort()[::-1]
        
        keep: list[int] = []
        while order.size > 0:
            i: int = order[0]
            keep.append(i)
            
            xx1: np.ndarray = np.maximum(x1[i], x1[order[1:]])
            yy1: np.ndarray = np.maximum(y1[i], y1[order[1:]])
            xx2: np.ndarray = np.minimum(x2[i], x2[order[1:]])
            yy2: np.ndarray = np.minimum(y2[i], y2[order[1:]])
            
            w: np.ndarray = np.maximum(0.0, xx2 - xx1)
            h: np.ndarray = np.maximum(0.0, yy2 - yy1)
            inter: np.ndarray = w * h
            
            iou: np.ndarray = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds: np.ndarray = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        keep_arr: np.ndarray = np.array(keep)
        nms_landmarks: Optional[np.ndarray] = landmarks[keep_arr] if landmarks is not None else None
        return boxes[keep_arr], scores[keep_arr], nms_landmarks
    
    def process(self, outputs: list[np.ndarray] | dict) -> list[dict]:
        """
        Process raw SCRFD network outputs.
        
        Args:
            outputs: Dict with layer names as keys from Hailo inference.
                     Expected: 9 outputs (3 scales × 3 types: bbox, score, landmarks)
        
        Returns:
            List of detection dicts with keys: bbox, confidence, center, landmarks
        """
        # Handle dict output from Hailo - sort by layer name to get consistent order
        if isinstance(outputs, dict):
            # Sort keys to ensure consistent ordering
            sorted_keys = sorted(outputs.keys())
            outputs = [outputs[k] for k in sorted_keys]
            logger.info(f"Sorted layer names: {sorted_keys}")
        
        if len(outputs) != 9:
            logger.warning(f"Expected 9 outputs, got {len(outputs)}")
            logger.warning(f"Output shapes: {[o.shape for o in outputs]}")
            return []
        
        # SCRFD outputs for 3 scales, each scale has 3 outputs:
        # - bbox prediction (4 channels)
        # - class scores (2 channels) 
        # - landmarks (10 channels)
        # Order after sorting should be: bbox0, bbox1, bbox2, kps0, kps1, kps2, score0, score1, score2
        
        # Group outputs by type based on channel count
        box_outputs: list[np.ndarray] = []
        score_outputs: list[np.ndarray] = []
        landmark_outputs: list[np.ndarray] = []
        
        for output in outputs:
            num_channels = output.shape[1] if len(output.shape) == 4 else output.shape[0]
            
            if num_channels == 4:
                # Bounding box output
                box_outputs.append(output)
            elif num_channels == 2:
                # Score output
                score_outputs.append(output)
            elif num_channels == 10:
                # Landmark output
                landmark_outputs.append(output)
            else:
                logger.warning(f"Unknown output shape: {output.shape}")
        
        if len(box_outputs) != 3 or len(score_outputs) != 3 or len(landmark_outputs) != 3:
            logger.error(f"Expected 3 of each output type, got: bbox={len(box_outputs)}, score={len(score_outputs)}, landmarks={len(landmark_outputs)}")
            return []
        
        # Process each scale
        all_box_preds: list[np.ndarray] = []
        all_score_preds: list[np.ndarray] = []
        all_landmark_preds: list[np.ndarray] = []
        
        for bbox, score, landmarks in zip(box_outputs, score_outputs, landmark_outputs):
            # Reshape from (1, C, H, W) to (N, C) where N = H*W
            if len(bbox.shape) == 4:
                _, c, h, w = bbox.shape
                bbox_flat = bbox.reshape(1, c, -1).transpose(0, 2, 1)[0]
                score_flat = score.reshape(1, 2, -1).transpose(0, 2, 1)[0]
                landmark_flat = landmarks.reshape(1, 10, -1).transpose(0, 2, 1)[0]
            else:
                bbox_flat = bbox
                score_flat = score
                landmark_flat = landmarks
            
            # Dequantize from uint8 to float32
            bbox_float = bbox_flat.astype(np.float32) / 32.0
            score_float = score_flat.astype(np.float32) / 255.0
            landmark_float = (landmark_flat.astype(np.float32) - 113.0) / 29.0
            
            all_box_preds.append(bbox_float)
            all_score_preds.append(score_float)
            all_landmark_preds.append(landmark_float)
        
        # Concatenate all scales
        all_boxes = np.concatenate(all_box_preds, axis=0)
        all_scores = np.concatenate(all_score_preds, axis=0)
        all_landmarks = np.concatenate(all_landmark_preds, axis=0)
        
        # Get face class scores (class 1, not background)
        face_scores = all_scores[:, 1]
        
        # Filter by confidence threshold
        mask = face_scores >= self.confidence_threshold
        filtered_boxes = all_boxes[mask]
        filtered_scores = face_scores[mask]
        filtered_landmarks = all_landmarks[mask]
        
        if len(filtered_boxes) == 0:
            return []
        
        # Decode predictions
        decoded_boxes = self._decode_boxes(box_preds=filtered_boxes)
        decoded_landmarks = self._decode_landmarks(landmark_preds=filtered_landmarks)
        
        # Apply NMS
        nms_boxes, nms_scores, nms_landmarks = self._nms(
            boxes=decoded_boxes, 
            scores=filtered_scores, 
            landmarks=decoded_landmarks
        )
        
        # Convert to detection format
        detections: list[dict] = []
        for i in range(len(nms_boxes)):
            x1, y1, x2, y2 = nms_boxes[i]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            detection = {
                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                'confidence': float(nms_scores[i]),
                'center': (center_x, center_y),
                'landmarks': nms_landmarks[i].tolist() if nms_landmarks is not None else []
            }
            detections.append(detection)
        
        return detections


# ============================================================================
# Face Selection and Tracking
# ============================================================================

def select_primary_face(
    faces: list[dict], 
    prev_center: Optional[tuple[float, float]] = None
) -> Optional[dict]:
    """Select face to track: closest to prev position or largest."""
    if not faces:
        return None
    
    if prev_center is not None:
        # Track closest face
        prev_x, prev_y = prev_center
        min_dist: float = float('inf')
        closest_face: Optional[dict] = None
        
        for face in faces:
            fx, fy = face['center']
            dist: float = np.sqrt((fx - prev_x)**2 + (fy - prev_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_face = face
        
        return closest_face
    else:
        # Select largest face
        max_area: float = 0.0
        largest_face: Optional[dict] = None
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            area: float = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_face = face
        
        return largest_face


def draw_visualization(
    frame: np.ndarray,
    face: Optional[dict],
    pan_angle: float,
    tilt_angle: float,
    fps: float,
    is_locked_x: bool = False,
    is_locked_y: bool = False,
    num_faces: int = 0
) -> np.ndarray:
    """Draw tracking visualization."""
    h, w = frame.shape[:2]
    center_x: int = w // 2
    center_y: int = h // 2
    
    # Center crosshair
    cv2.line(img=frame, pt1=(center_x, 0), pt2=(center_x, h), color=(255, 255, 255), thickness=2)
    cv2.line(img=frame, pt1=(0, center_y), pt2=(w, center_y), color=(255, 255, 255), thickness=2)
    
    # Deadzone box
    dz_l: int = center_x - DEADZONE_PIXELS
    dz_r: int = center_x + DEADZONE_PIXELS
    dz_t: int = center_y - DEADZONE_PIXELS
    dz_b: int = center_y + DEADZONE_PIXELS
    cv2.rectangle(img=frame, pt1=(dz_l, dz_t), pt2=(dz_r, dz_b), color=(128, 128, 128), thickness=1)
    
    if face is not None:
        # Face bounding box
        x1, y1, x2, y2 = face['bbox']
        box_color: tuple[int, int, int] = (0, 255, 0) if (is_locked_x and is_locked_y) else (255, 165, 0)
        cv2.rectangle(
            img=frame,
            pt1=(int(x1), int(y1)),
            pt2=(int(x2), int(y2)),
            color=box_color,
            thickness=2
        )
        
        # Face center
        fx, fy = face['center']
        cv2.circle(
            img=frame,
            center=(int(fx), int(fy)),
            radius=8,
            color=box_color,
            thickness=-1
        )
        
        # Landmarks
        landmarks: list = face.get('landmarks', [])
        if landmarks and len(landmarks) >= 10:
            for i in range(0, 10, 2):
                lx: int = int(landmarks[i])
                ly: int = int(landmarks[i + 1])
                cv2.circle(img=frame, center=(lx, ly), radius=3, color=(0, 255, 255), thickness=-1)
        
        # Tracking lines
        line_color_x: tuple[int, int, int] = (0, 255, 0) if is_locked_x else (255, 255, 0)
        line_color_y: tuple[int, int, int] = (0, 255, 0) if is_locked_y else (255, 255, 0)
        cv2.line(img=frame, pt1=(center_x, int(fy)), pt2=(int(fx), int(fy)), color=line_color_x, thickness=2)
        cv2.line(img=frame, pt1=(int(fx), center_y), pt2=(int(fx), int(fy)), color=line_color_y, thickness=2)
        
        # Confidence
        cv2.putText(
            img=frame,
            text=f"Conf: {face['confidence']:.2f}",
            org=(int(x1), int(y1) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=box_color,
            thickness=2
        )
    
    # Status text
    status: str = "LOCKED" if (is_locked_x and is_locked_y and face) else ("TRACKING" if face else "SEARCHING")
    status_color: tuple[int, int, int] = (0, 255, 0) if status == "LOCKED" else ((255, 165, 0) if status == "TRACKING" else (255, 0, 0))
    
    cv2.putText(img=frame, text=status, org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=status_color, thickness=2)
    cv2.putText(img=frame, text=f"Faces: {num_faces}", org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    cv2.putText(img=frame, text=f"Pan: {pan_angle:.1f}° {'✓' if is_locked_x else ''}", org=(10, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    cv2.putText(img=frame, text=f"Tilt: {tilt_angle:.1f}° {'✓' if is_locked_y else ''}", org=(10, 115), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    cv2.putText(img=frame, text=f"FPS: {fps:.1f}", org=(10, 140), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    
    if face:
        fx, fy = face['center']
        error_x: int = int(fx - center_x)
        error_y: int = int(fy - center_y)
        cv2.putText(img=frame, text=f"Error: X={error_x:+4d}px Y={error_y:+4d}px", org=(10, 165), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
    
    return frame


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run face tracking system."""
    logger.info("="*60)
    logger.info("HAILO AI FACE TRACKER (FIXED)")
    logger.info(f"Model: {FACE_MODEL}")
    logger.info("="*60)
    
    # Initialize servos
    kit: ServoKit = ServoKit(channels=16)
    pan_angle: float = 90.0
    tilt_angle: float = 90.0
    kit.servo[PAN_SERVO_CHANNEL].angle = pan_angle
    kit.servo[TILT_SERVO_CHANNEL].angle = tilt_angle
    time.sleep(0.5)
    
    # Test servos
    logger.info("Testing servos...")
    for angle in [60.0, 120.0, 90.0]:
        kit.servo[PAN_SERVO_CHANNEL].angle = angle
        time.sleep(0.3)
    for angle in [60.0, 120.0, 90.0]:
        kit.servo[TILT_SERVO_CHANNEL].angle = angle
        time.sleep(0.3)
    logger.info("Servos OK!")
    
    # Initialize Hailo
    logger.info("Initializing Hailo...")
    try:
        with Hailo(MODEL_PATH) as hailo:
            model_h: int
            model_w: int
            model_h, model_w, _ = hailo.get_input_shape()
            logger.info(f"Model input: {model_w}x{model_h}")
            
            # Initialize post-processor
            post_processor: SCRFDPostProcessor = SCRFDPostProcessor(
                model_input_size=(model_h, model_w),
                confidence_threshold=CONFIDENCE_THRESHOLD
            )
            
            # Initialize camera
            logger.info("Starting camera...")
            picam2: Picamera2 = Picamera2()
            main_config: dict = {'size': (CAMERA_WIDTH, CAMERA_HEIGHT), 'format': 'RGB888'}
            lores_config: dict = {'size': (model_w, model_h), 'format': 'RGB888'}
            config: dict = picam2.create_preview_configuration(main=main_config, lores=lores_config)
            picam2.configure(config)
            picam2.start()
            time.sleep(1)
            
            logger.info(f"Confidence: {CONFIDENCE_THRESHOLD}")
            logger.info(f"P-gain: {PROPORTIONAL_GAIN}")
            logger.info(f"Deadzone: ±{DEADZONE_PIXELS}px")
            logger.info("Press 'q' to quit")
            logger.info("="*60)
            
            # Tracking state
            frame_count: int = 0
            last_fps_time: float = time.time()
            fps: float = 0.0
            prev_face_center: Optional[tuple[float, float]] = None
            
            try:
                while True:
                    # Capture frames
                    main_frame: np.ndarray = picam2.capture_array(name='main')
                    lores_frame: np.ndarray = picam2.capture_array(name='lores')
                    
                    # Run inference
                    raw_outputs: list[np.ndarray] | dict = hailo.run(lores_frame)
                    
                    # Debug: inspect output format on first frame
                    if frame_count == 0:
                        logger.info(f"Raw output type: {type(raw_outputs)}")
                        if isinstance(raw_outputs, dict):
                            logger.info(f"Output keys: {list(raw_outputs.keys())}")
                            logger.info(f"Number of outputs: {len(raw_outputs)}")
                            for k, v in list(raw_outputs.items())[:3]:  # Show first 3
                                logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    
                    # Post-process to get detections
                    faces: list[dict] = post_processor.process(outputs=raw_outputs)
                    
                    # Select primary face
                    primary_face: Optional[dict] = select_primary_face(faces=faces, prev_center=prev_face_center)
                    
                    is_locked_x: bool = False
                    is_locked_y: bool = False
                    
                    if primary_face is not None:
                        # Scale coordinates from model to camera resolution
                        scale_x: float = CAMERA_WIDTH / model_w
                        scale_y: float = CAMERA_HEIGHT / model_h
                        
                        face_x, face_y = primary_face['center']
                        face_x *= scale_x
                        face_y *= scale_y
                        
                        prev_face_center = (face_x, face_y)
                        
                        # Calculate errors
                        center_x: int = CAMERA_WIDTH // 2
                        center_y: int = CAMERA_HEIGHT // 2
                        error_x: float = face_x - center_x
                        error_y: float = face_y - center_y
                        
                        # Check lock status
                        is_locked_x = abs(error_x) <= DEADZONE_PIXELS
                        is_locked_y = abs(error_y) <= DEADZONE_PIXELS
                        
                        # Pan control
                        if not is_locked_x:
                            pan_angle += error_x * PROPORTIONAL_GAIN
                            pan_angle = np.clip(pan_angle, PAN_MIN, PAN_MAX)
                            kit.servo[PAN_SERVO_CHANNEL].angle = pan_angle
                        
                        # Tilt control
                        if not is_locked_y:
                            tilt_angle += error_y * PROPORTIONAL_GAIN
                            tilt_angle = np.clip(tilt_angle, TILT_MIN, TILT_MAX)
                            kit.servo[TILT_SERVO_CHANNEL].angle = tilt_angle
                        
                        # Scale bbox and landmarks for display
                        x1, y1, x2, y2 = primary_face['bbox']
                        primary_face['bbox'] = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                        primary_face['center'] = (face_x, face_y)
                        
                        if primary_face['landmarks']:
                            scaled_landmarks: list[float] = []
                            for i, val in enumerate(primary_face['landmarks']):
                                scale: float = scale_x if i % 2 == 0 else scale_y
                                scaled_landmarks.append(val * scale)
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
                    vis_frame: np.ndarray = draw_visualization(
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
                # Reset servos
                logger.info("Resetting servos...")
                kit.servo[PAN_SERVO_CHANNEL].angle = 90.0
                kit.servo[TILT_SERVO_CHANNEL].angle = 90.0
                time.sleep(0.5)
                kit.servo[PAN_SERVO_CHANNEL].angle = None
                kit.servo[TILT_SERVO_CHANNEL].angle = None
                
                picam2.stop()
                cv2.destroyAllWindows()
                logger.info("Done!")
    
    except Exception as e:
        logger.error(msg=f"Error: {e}")
        logger.error(msg=f"Check model path: {MODEL_PATH}")
        raise


if __name__ == "__main__":
    main()