#!/usr/bin/env python3
"""
Hailo AI HAT+ Face Landmark Detection with Mouth State Detection
Detects 68 facial landmarks and determines if mouth is open or closed
"""

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType
)
from typing import NamedTuple


class FaceBox(NamedTuple):
    """Detected face bounding box"""
    x: int
    y: int
    w: int
    h: int
    confidence: float


class FaceLandmarks(NamedTuple):
    """68-point facial landmarks"""
    points: np.ndarray  # Shape: (68, 2)
    mouth_open: bool
    mouth_aspect_ratio: float


class HailoFaceLandmarkDetector:
    """Face landmark detection using Hailo AI HAT+"""
    
    def __init__(
        self,
        face_detection_hef: str = "/usr/share/hailo-models/scrfd_2.5g.hef",
        landmark_hef: str = "/usr/share/hailo-models/tddfa_mobilenet_v1.hef",
        mouth_threshold: float = 0.6
    ) -> None:
        """
        Initialize Hailo face landmark detector
        
        Args:
            face_detection_hef: Path to face detection HEF model
            landmark_hef: Path to facial landmark HEF model  
            mouth_threshold: Threshold for mouth open detection (MAR > threshold = open)
        """
        self.mouth_threshold = mouth_threshold
        
        # Initialize Hailo device
        self.device = VDevice()
        
        # Load face detection model
        print(f"Loading face detection model: {face_detection_hef}")
        self.face_hef = HEF(face_detection_hef)
        
        # Load landmark detection model
        print(f"Loading landmark model: {landmark_hef}")
        self.landmark_hef = HEF(landmark_hef)
        
        # Configure models
        self._configure_models()
        
        print("Hailo Face Landmark Detector initialized")
    
    def _configure_models(self) -> None:
        """Configure Hailo models"""
        # Configure face detection
        configure_params = ConfigureParams.create_from_hef(
            hef=self.face_hef,
            interface=HailoStreamInterface.PCIe
        )
        self.face_network_group = self.device.configure(self.face_hef, configure_params)[0]
        
        # Configure landmark detection
        configure_params = ConfigureParams.create_from_hef(
            hef=self.landmark_hef,
            interface=HailoStreamInterface.PCIe
        )
        self.landmark_network_group = self.device.configure(self.landmark_hef, configure_params)[0]
    
    def detect_faces(self, image: np.ndarray) -> list[FaceBox]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected face bounding boxes
        """
        # Preprocess image for face detection
        input_height, input_width = 640, 640
        resized = cv2.resize(image, (input_width, input_height))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        with InferVStreams(self.face_network_group, 
                          InputVStreamParams.make_from_network_group(self.face_network_group, quantized=False, format_type=FormatType.FLOAT32),
                          OutputVStreamParams.make_from_network_group(self.face_network_group, quantized=False, format_type=FormatType.FLOAT32)) as infer_pipeline:
            
            input_dict = {list(infer_pipeline.input_vstreams.keys())[0]: input_data}
            output = infer_pipeline.infer(input_dict)
        
        # Parse face detections (this depends on model output format)
        faces = self._parse_face_detections(output, image.shape)
        return faces
    
    def _parse_face_detections(self, output: dict, image_shape: tuple[int, int, int]) -> list[FaceBox]:
        """Parse face detection output"""
        # This is model-specific - adjust based on your face detection model output format
        # Placeholder implementation
        faces = []
        
        # Example: assuming output contains bboxes and scores
        # You'll need to adjust this based on actual model output
        output_key = list(output.keys())[0]
        detections = output[output_key]
        
        height, width = image_shape[:2]
        
        # Parse detections (format depends on specific model)
        # This is a simplified example
        for detection in detections:
            if len(detection) >= 5:
                x, y, w, h, conf = detection[:5]
                if conf > 0.5:  # Confidence threshold
                    faces.append(FaceBox(
                        x=int(x * width),
                        y=int(y * height),
                        w=int(w * width),
                        h=int(h * height),
                        confidence=float(conf)
                    ))
        
        return faces
    
    def detect_landmarks(self, image: np.ndarray, face: FaceBox) -> FaceLandmarks:
        """
        Detect 68 facial landmarks for a face
        
        Args:
            image: Input image (BGR format)
            face: Face bounding box
            
        Returns:
            68-point facial landmarks with mouth state
        """
        # Crop and preprocess face
        face_crop = image[face.y:face.y+face.h, face.x:face.x+face.w]
        
        # Resize to model input size (120x120 for tddfa_mobilenet_v1)
        input_size = 120
        resized = cv2.resize(face_crop, (input_size, input_size))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        with InferVStreams(self.landmark_network_group,
                          InputVStreamParams.make_from_network_group(self.landmark_network_group, quantized=False, format_type=FormatType.FLOAT32),
                          OutputVStreamParams.make_from_network_group(self.landmark_network_group, quantized=False, format_type=FormatType.FLOAT32)) as infer_pipeline:
            
            input_dict = {list(infer_pipeline.input_vstreams.keys())[0]: input_data}
            output = infer_pipeline.infer(input_dict)
        
        # Parse landmarks
        landmarks = self._parse_landmarks(output, face, face_crop.shape)
        
        # Calculate mouth state
        mouth_open, mar = self._calculate_mouth_state(landmarks.points)
        
        return FaceLandmarks(
            points=landmarks.points,
            mouth_open=mouth_open,
            mouth_aspect_ratio=mar
        )
    
    def _parse_landmarks(
        self, 
        output: dict, 
        face: FaceBox,
        crop_shape: tuple[int, int, int]
    ) -> FaceLandmarks:
        """Parse landmark detection output"""
        # Get output tensor (format depends on model)
        output_key = list(output.keys())[0]
        landmarks_raw = output[output_key][0]  # Remove batch dimension
        
        # Reshape to (68, 2) if needed
        if landmarks_raw.shape[-1] == 136:  # Flattened 68 * 2
            landmarks_raw = landmarks_raw.reshape(68, 2)
        
        # Scale landmarks back to original face coordinates
        h, w = crop_shape[:2]
        landmarks = landmarks_raw.copy()
        landmarks[:, 0] = landmarks[:, 0] * w + face.x
        landmarks[:, 1] = landmarks[:, 1] * h + face.y
        
        return FaceLandmarks(
            points=landmarks,
            mouth_open=False,
            mouth_aspect_ratio=0.0
        )
    
    def _calculate_mouth_state(self, landmarks: np.ndarray) -> tuple[bool, float]:
        """
        Calculate if mouth is open using Mouth Aspect Ratio (MAR)
        
        Mouth landmarks (iBUG 68-point format):
        - Outer lips: 48-59
        - Inner lips: 60-67
        
        Args:
            landmarks: 68 facial landmarks
            
        Returns:
            (is_mouth_open, mouth_aspect_ratio)
        """
        # Get mouth landmarks (points 48-67)
        # Outer lip landmarks
        left_corner = landmarks[48]   # Left corner
        right_corner = landmarks[54]  # Right corner
        top_outer = landmarks[51]     # Top center outer
        bottom_outer = landmarks[57]  # Bottom center outer
        
        # Inner lip landmarks for better accuracy
        top_inner = landmarks[62]     # Top center inner
        bottom_inner = landmarks[66]  # Bottom center inner
        
        # Calculate distances
        width = np.linalg.norm(right_corner - left_corner)
        
        # Vertical distances (use inner lips for mouth opening)
        height_outer = np.linalg.norm(top_outer - bottom_outer)
        height_inner = np.linalg.norm(top_inner - bottom_inner)
        
        # Average vertical distance
        height = (height_outer + height_inner) / 2.0
        
        # Mouth Aspect Ratio (MAR)
        mar = height / (width + 1e-6)
        
        # Determine if mouth is open
        is_open = mar > self.mouth_threshold
        
        return is_open, mar
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[FaceLandmarks]]:
        """
        Process a single frame: detect faces and landmarks
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            (annotated_frame, list_of_landmarks)
        """
        annotated = frame.copy()
        all_landmarks = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Process each face
        for face in faces:
            # Draw face bounding box
            cv2.rectangle(
                annotated,
                (face.x, face.y),
                (face.x + face.w, face.y + face.h),
                (0, 255, 0),
                thickness=2
            )
            
            # Detect landmarks
            landmarks = self.detect_landmarks(frame, face)
            all_landmarks.append(landmarks)
            
            # Draw landmarks
            for i, (x, y) in enumerate(landmarks.points.astype(int)):
                # Color mouth landmarks differently
                if 48 <= i < 68:  # Mouth region
                    color = (0, 0, 255) if landmarks.mouth_open else (255, 0, 0)
                    radius = 3
                else:
                    color = (0, 255, 255)
                    radius = 2
                
                cv2.circle(annotated, (x, y), radius=radius, color=color, thickness=-1)
            
            # Draw mouth state text
            mouth_text = f"Mouth: {'OPEN' if landmarks.mouth_open else 'CLOSED'} ({landmarks.mouth_aspect_ratio:.2f})"
            cv2.putText(
                annotated,
                mouth_text,
                (face.x, face.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=2
            )
        
        return annotated, all_landmarks
    
    def close(self) -> None:
        """Release resources"""
        self.face_network_group.release()
        self.landmark_network_group.release()
        print("Detector closed")


def main() -> None:
    """Main function to run face landmark detection"""
    
    # Initialize detector
    detector = HailoFaceLandmarkDetector(
        face_detection_hef="/usr/share/hailo-models/scrfd_2.5g.hef",
        landmark_hef="/usr/share/hailo-models/tddfa_mobilenet_v1.hef",
        mouth_threshold=0.6
    )
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                raise RuntimeError("Failed to read frame from camera")
            
            # Process frame
            annotated, landmarks = detector.process_frame(frame)
            
            # Display
            cv2.imshow("Hailo Face Landmarks - Mouth Detection", annotated)
            
            # Print mouth states
            for i, lm in enumerate(landmarks):
                status = "OPEN" if lm.mouth_open else "CLOSED"
                print(f"Face {i}: Mouth {status} (MAR: {lm.mouth_aspect_ratio:.3f})")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
