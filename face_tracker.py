#!/usr/bin/env python3
"""
Raspberry Pi 5 AI Camera with Servo Face Tracking
Detects people and moves servos to center the first detected person in frame.
"""

import time
from typing import Any
import numpy as np
import cv2

from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from adafruit_servokit import ServoKit


class Detection:
    """Represents a single detected object."""
    
    def __init__(
        self,
        *,
        coords: tuple[float, float, float, float],
        category: int,
        confidence: float,
        metadata: dict[str, Any]
    ) -> None:
        """
        Initialize a Detection object.
        
        Args:
            coords: Bounding box coordinates (x, y, w, h)
            category: Category/class index
            confidence: Confidence score (0-1)
            metadata: Frame metadata from camera
        """
        self.category: int = category
        self.confidence: float = confidence
        self.box: tuple[int, int, int, int] = imx500.convert_inference_coords(
            coords=coords,
            metadata=metadata,
            picam2=picam2
        )


class ServoController:
    """Controls servos to track detected faces."""
    
    def __init__(
        self,
        *,
        pitch_channel: int = 3,
        yaw_channel: int = 7,
        roll_channel: int = 11
    ) -> None:
        """
        Initialize servo controller.
        
        Args:
            pitch_channel: Servo channel for pitch (vertical movement)
            yaw_channel: Servo channel for yaw (horizontal movement)
            roll_channel: Servo channel for roll (not used for tracking)
        """
        print("Initializing servo controller...")
        time.sleep(2)  # Wait for servo board to initialize
        
        self.kit: ServoKit = ServoKit(channels=16)
        self.pitch_channel: int = pitch_channel
        self.yaw_channel: int = yaw_channel
        self.roll_channel: int = roll_channel
        
        # Current servo angles (start at center)
        self.pitch_angle: float = 90.0
        self.yaw_angle: float = 90.0
        self.roll_angle: float = 90.0
        
        # Servo limits (safety margins)
        self.min_angle: float = 30.0
        self.max_angle: float = 150.0
        
        # Control parameters
        self.p_gain: float = 0.08  # Proportional gain - adjust for sensitivity
        self.deadzone: float = 20.0  # Pixels - ignore small errors
        
        # Center the servos initially
        self.center_servos()
        
        print("Servo controller initialized!")
    
    def center_servos(self) -> None:
        """Move all servos to center position."""
        print("Centering servos...")
        self.pitch_angle = 90.0
        self.yaw_angle = 90.0
        self.roll_angle = 90.0
        
        self.kit.servo[self.pitch_channel].angle = self.pitch_angle
        self.kit.servo[self.yaw_channel].angle = self.yaw_angle
        self.kit.servo[self.roll_channel].angle = self.roll_angle
        time.sleep(1)
    
    def clamp_angle(self, *, angle: float) -> float:
        """
        Clamp angle to safe servo range.
        
        Args:
            angle: Desired angle
            
        Returns:
            Clamped angle within safe limits
        """
        return max(self.min_angle, min(self.max_angle, angle))
    
    def track_target(
        self,
        *,
        target_x: int,
        target_y: int,
        frame_width: int,
        frame_height: int
    ) -> None:
        """
        Adjust servos to center the target in frame.
        
        Args:
            target_x: X coordinate of target center
            target_y: Y coordinate of target center
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        # Calculate center of frame
        center_x: float = frame_width / 2.0
        center_y: float = frame_height / 2.0
        
        # Calculate error (how far target is from center)
        error_x: float = target_x - center_x
        error_y: float = target_y - center_y
        
        # Apply deadzone - don't move for small errors
        if abs(error_x) < self.deadzone and abs(error_y) < self.deadzone:
            return
        
        # Calculate servo adjustments (proportional control)
        # Positive error_x means target is right of center -> decrease yaw
        # Positive error_y means target is below center -> increase pitch
        yaw_adjustment: float = -error_x * self.p_gain
        pitch_adjustment: float = error_y * self.p_gain
        
        # Update angles
        self.yaw_angle = self.clamp_angle(angle=self.yaw_angle + yaw_adjustment)
        self.pitch_angle = self.clamp_angle(angle=self.pitch_angle + pitch_adjustment)
        
        # Apply to servos
        self.kit.servo[self.yaw_channel].angle = self.yaw_angle
        self.kit.servo[self.pitch_channel].angle = self.pitch_angle


def parse_detections(*, metadata: dict[str, Any]) -> list[Detection]:
    """
    Parse neural network output tensor into Detection objects.
    
    Args:
        metadata: Frame metadata containing inference results
        
    Returns:
        List of Detection objects above confidence threshold
    """
    global last_detections
    
    # Get network outputs from metadata
    np_outputs: np.ndarray | None = imx500.get_outputs(
        metadata=metadata,
        add_batch=True
    )
    
    if np_outputs is None:
        return last_detections
    
    input_w, input_h = imx500.get_input_size()
    
    # Extract boxes, scores, and classes from network output
    boxes: np.ndarray = np_outputs[0][0]
    scores: np.ndarray = np_outputs[1][0]
    classes: np.ndarray = np_outputs[2][0]
    
    # Normalize boxes if needed
    if intrinsics.bbox_normalization:
        boxes = boxes / input_h
    
    # Reorder boxes if needed (some models use xy, others yx)
    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]
    
    # Split boxes into individual coordinates
    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)
    
    # Filter detections by confidence threshold and create Detection objects
    detections: list[Detection] = [
        Detection(
            coords=box,
            category=int(category),
            confidence=float(score),
            metadata=metadata
        )
        for box, score, category in zip(boxes, scores, classes)
        if score > CONFIDENCE_THRESHOLD
    ]
    
    last_detections = detections
    return detections


def draw_detections(request, stream: str = "main") -> None:
    """
    Draw bounding boxes and labels on the camera preview.
    
    Args:
        request: Camera request with frame data
        stream: Stream name to draw on
    """
    detections = last_detections
    if not detections:
        return
    
    with MappedArray(request, stream) as m:
        frame_height, frame_width = m.array.shape[:2]
        
        # Draw crosshair at center
        center_x: int = frame_width // 2
        center_y: int = frame_height // 2
        cv2.drawMarker(
            img=m.array,
            position=(center_x, center_y),
            color=(255, 0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2
        )
        
        for idx, detection in enumerate(detections):
            x, y, w, h = detection.box
            label: str = labels[detection.category]
            
            # Highlight first person detection differently
            is_target: bool = (idx == 0 and label.lower() == "person")
            color: tuple[int, int, int, int] = (0, 0, 255, 255) if is_target else (0, 255, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(
                img=m.array,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=color,
                thickness=3 if is_target else 2
            )
            
            # Draw center point of detection
            if is_target:
                det_center_x: int = x + w // 2
                det_center_y: int = y + h // 2
                cv2.circle(
                    img=m.array,
                    center=(det_center_x, det_center_y),
                    radius=5,
                    color=(0, 0, 255, 255),
                    thickness=-1
                )
            
            # Prepare label text
            text: str = f"{label} {detection.confidence:.2f}"
            if is_target:
                text = f"TRACKING: {text}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text=text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1
            )
            
            # Draw text background
            cv2.rectangle(
                img=m.array,
                pt1=(x, y - text_height - 10),
                pt2=(x + text_width + 10, y),
                color=color,
                thickness=-1
            )
            
            # Draw text
            cv2.putText(
                img=m.array,
                text=text,
                org=(x + 5, y - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255, 255),
                thickness=1
            )


def print_detections(*, detections: list[Detection], frame_number: int) -> None:
    """
    Print detected objects to console.
    
    Args:
        detections: List of Detection objects
        frame_number: Current frame number
    """
    if not detections:
        if frame_number % 30 == 0:  # Print every 30 frames to avoid spam
            print(f"Frame {frame_number}: No objects detected")
        return
    
    # Only print if we detect a person
    people: list[Detection] = [d for d in detections if labels[d.category].lower() == "person"]
    if not people:
        return
    
    print(f"\n{'='*60}")
    print(f"Frame {frame_number} - Tracking {len(people)} person(s)")
    print(f"{'='*60}")
    
    for idx, detection in enumerate(people, start=1):
        x, y, w, h = detection.box
        center_x: int = x + w // 2
        center_y: int = y + h // 2
        
        print(f"  {idx}. Person")
        print(f"     Confidence: {detection.confidence:.2%}")
        print(f"     Center: ({center_x}, {center_y})")
        print(f"     Box: x={x}, y={y}, w={w}, h={h}")
        
        if idx == 1:
            print(f"     >>> TRACKING THIS PERSON <<<")


def main() -> None:
    """Main loop - continuously capture, detect, and track faces."""
    global picam2, imx500, intrinsics, labels, last_detections, servo_controller
    
    print("="*60)
    print("Initializing Face Tracking System...")
    print("="*60)
    
    # Initialize servo controller first
    servo_controller = ServoController(
        pitch_channel=3,
        yaw_channel=7,
        roll_channel=11
    )
    
    print(f"\nLoading AI model: {MODEL_PATH}")
    
    # Initialize IMX500 device with neural network model
    imx500 = IMX500(network_file=MODEL_PATH)
    
    # Get network intrinsics (model configuration)
    intrinsics = imx500.network_intrinsics
    
    # Load class labels
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = f.read().splitlines()
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {LABELS_PATH}")
        print("Using numeric class indices instead")
        labels = [f"Class {i}" for i in range(1000)]
    
    # Update intrinsics with defaults
    intrinsics.update_with_defaults()
    
    # Initialize Picamera2
    picam2 = Picamera2(camera_num=imx500.camera_num)
    
    # Create camera configuration with preview
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12
    )
    
    # Show firmware loading progress
    print("\nLoading neural network firmware onto IMX500 sensor...")
    print("(This may take 1-2 minutes on first run)")
    imx500.show_network_fw_progress_bar()
    
    # Start camera with preview window
    picam2.start(config=config, show_preview=True)
    
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    
    print("\n" + "="*60)
    print("FACE TRACKING ACTIVE!")
    print("- Detects people and centers first person in frame")
    print("- Red box = tracked target")
    print("- Red crosshair = frame center")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    frame_count: int = 0
    
    # Register callback to draw detections on each frame
    picam2.pre_callback = draw_detections
    
    try:
        while True:
            # Capture frame with metadata
            request = picam2.capture_request()
            
            try:
                # Get frame metadata
                metadata: dict[str, Any] = request.get_metadata()
                
                # Get frame dimensions
                frame_shape = request.make_array("main").shape
                frame_height: int = frame_shape[0]
                frame_width: int = frame_shape[1]
                
                # Parse detections from neural network output
                detections: list[Detection] = parse_detections(metadata=metadata)
                
                # Find first person detection
                people: list[Detection] = [
                    d for d in detections 
                    if labels[d.category].lower() == "person"
                ]
                
                if people:
                    # Track the first detected person
                    target: Detection = people[0]
                    x, y, w, h = target.box
                    
                    # Calculate center of bounding box
                    target_center_x: int = x + w // 2
                    target_center_y: int = y + h // 2
                    
                    # Update servo positions to track target
                    servo_controller.track_target(
                        target_x=target_center_x,
                        target_y=target_center_y,
                        frame_width=frame_width,
                        frame_height=frame_height
                    )
                
                # Print detection info periodically
                if people or frame_count % 30 == 0:
                    print_detections(detections=detections, frame_number=frame_count)
                
                frame_count += 1
                
                # Small delay
                time.sleep(0.033)  # ~30 fps
                
            finally:
                # Always release the request
                request.release()
                
    except KeyboardInterrupt:
        print("\n\nStopping face tracking system...")
    
    finally:
        # Return servos to center position
        servo_controller.center_servos()
        
        # Clean up camera
        picam2.stop()
        print("System stopped. Goodbye!")


# Configuration
MODEL_PATH: str = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
LABELS_PATH: str = "/home/pi/picamera2/examples/imx500/assets/coco_labels.txt"
CONFIDENCE_THRESHOLD: float = 0.55

# Global state
picam2 = None
imx500 = None
intrinsics = None
labels = []
last_detections = []
servo_controller = None


if __name__ == "__main__":
    main()