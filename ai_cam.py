#!/usr/bin/env python3
"""
Raspberry Pi 5 AI Camera - Object Detection Loop
Continuously captures frames and prints detected objects using the IMX500 sensor.
"""

import time
from typing import Any
import numpy as np

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics


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
    # Format depends on model - this works for MobileNet SSD and YOLO models
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


def print_detections(*, detections: list[Detection], frame_number: int) -> None:
    """
    Print detected objects to console.
    
    Args:
        detections: List of Detection objects
        frame_number: Current frame number
    """
    if not detections:
        print(f"Frame {frame_number}: No objects detected")
        return
    
    print(f"\n{'='*60}")
    print(f"Frame {frame_number} - Detected {len(detections)} object(s):")
    print(f"{'='*60}")
    
    for idx, detection in enumerate(detections, start=1):
        label: str = labels[detection.category]
        x, y, w, h = detection.box
        
        print(f"  {idx}. {label}")
        print(f"     Confidence: {detection.confidence:.2%}")
        print(f"     Position: x={x}, y={y}, width={w}, height={h}")


def main() -> None:
    """Main loop - continuously capture and detect objects."""
    global picam2, imx500, intrinsics, labels, last_detections
    
    print("Initializing Raspberry Pi 5 AI Camera...")
    print(f"Loading model: {MODEL_PATH}")
    
    # Initialize IMX500 device with neural network model
    imx500 = IMX500(network_file=MODEL_PATH)
    
    # Get network intrinsics (model configuration)
    intrinsics: NetworkIntrinsics = imx500.network_intrinsics
    
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
    
    # Create camera configuration
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12
    )
    
    # Show firmware loading progress
    print("\nLoading neural network firmware onto IMX500 sensor...")
    print("(This may take 1-2 minutes on first run)")
    imx500.show_network_fw_progress_bar()
    
    # Start camera
    picam2.start(config=config, show_preview=False)
    
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    
    print("\n" + "="*60)
    print("AI Camera running! Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    frame_count: int = 0
    
    try:
        while True:
            # Capture frame with metadata
            request = picam2.capture_request()
            
            try:
                # Get frame metadata
                metadata: dict[str, Any] = request.get_metadata()
                
                # Parse detections from neural network output
                detections: list[Detection] = parse_detections(metadata=metadata)
                
                # Print what was detected
                print_detections(detections=detections, frame_number=frame_count)
                
                frame_count += 1
                
                # Small delay to make output readable
                time.sleep(0.1)
                
            finally:
                # Always release the request
                request.release()
                
    except KeyboardInterrupt:
        print("\n\nStopping camera...")
    
    finally:
        # Clean up
        picam2.stop()
        print("Camera stopped. Goodbye!")


# Configuration
MODEL_PATH: str = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
LABELS_PATH: str = "/home/pi/picamera2/examples/imx500/assets/coco_labels.txt"
CONFIDENCE_THRESHOLD: float = 0.55

# Global state
picam2: Picamera2 | None = None
imx500: IMX500 | None = None
intrinsics: NetworkIntrinsics | None = None
labels: list[str] = []
last_detections: list[Detection] = []


if __name__ == "__main__":
    main()