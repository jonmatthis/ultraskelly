#!/usr/bin/env python3
"""
Hailo AI HAT+ Face Landmark Detection with Mouth State Detection
Detects 68 facial landmarks and determines if mouth is open or closed

Auto-downloads models from Hailo Model Zoo if not found locally.

Requirements:
    - Hailo AI HAT+ (Hailo-8L or Hailo-8)
    - Raspberry Pi 5 or compatible device
    - hailo-all package installed
    - OpenCV and NumPy

Installation:
    sudo apt install -y hailo-all python3-opencv
    pip install numpy

Note: Models are downloaded to ~/.cache/hailo-models/
      First run will download ~50MB of models
"""

import cv2
import numpy as np
import urllib.request
import os
from pathlib import Path
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

    # Model download URLs from Hailo Model Zoo S3 bucket
    # Note: These are for Hailo-8L. Hailo-8 models may differ.
    MODEL_URLS = {
        "scrfd_2.5g": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/scrfd_2.5g.hef",
        "scrfd_2.5g_hailo8": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/scrfd_2.5g.hef",
        "tddfa_mobilenet_v1": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.0.0/tddfa_mobilenet_v1.hef",
    }

    def __init__(
            self,
            face_detection_hef: str = "/usr/share/hailo-models/scrfd_2.5g.hef",
            landmark_hef: str = "/usr/share/hailo-models/tddfa_mobilenet_v1.hef",
            mouth_threshold: float = 0.6,
            model_cache_dir: str = "~/.cache/hailo-models"
    ) -> None:
        """
        Initialize Hailo face landmark detector

        Args:
            face_detection_hef: Path to face detection HEF model
            landmark_hef: Path to facial landmark HEF model
            mouth_threshold: Threshold for mouth open detection (MAR > threshold = open)
            model_cache_dir: Directory to cache downloaded models
        """
        self.mouth_threshold = mouth_threshold
        self.model_cache_dir = Path(model_cache_dir).expanduser()

        # Ensure models exist (download if needed)
        face_detection_hef = self._ensure_model(face_detection_hef, "scrfd_2.5g")
        landmark_hef = self._ensure_model(landmark_hef, "tddfa_mobilenet_v1")

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

    def _ensure_model(self, model_path: str, model_name: str) -> str:
        """
        Ensure model exists, download if necessary

        Args:
            model_path: Preferred model path
            model_name: Model identifier for download

        Returns:
            Path to model file
        """
        # Check if model exists at specified path
        if os.path.exists(model_path):
            print(f"✓ Found model at: {model_path}")
            return model_path

        # Try cache directory
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.model_cache_dir / f"{model_name}.hef"

        if cache_path.exists():
            print(f"✓ Found cached model at: {cache_path}")
            return str(cache_path)

        # Download model
        if model_name not in self.MODEL_URLS:
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {model_path} and no download URL available"
            )

        url = self.MODEL_URLS[model_name]
        print(f"⬇ Downloading {model_name} from Hailo Model Zoo...")
        print(f"  URL: {url}")
        print(f"  Destination: {cache_path}")

        try:
            # Download with progress
            def report_progress(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')

            urllib.request.urlretrieve(url, cache_path, reporthook=report_progress)
            print(f"\n✓ Successfully downloaded {model_name}")
            return str(cache_path)

        except Exception as e:
            # Clean up partial download
            if cache_path.exists():
                cache_path.unlink()
            raise RuntimeError(f"Failed to download model {model_name}: {e}")

    def _configure_models(self) -> None:
        """Configure Hailo models"""
        # Configure face detection
        self.face_network_group = self.device.configure(
            self.face_hef,
            ConfigureParams.create_from_hef(self.face_hef, interface=HailoStreamInterface.PCIe)
        )[0]

        # Configure landmark detection
        self.landmark_network_group = self.device.configure(
            self.landmark_hef,
            ConfigureParams.create_from_hef(self.landmark_hef, interface=HailoStreamInterface.PCIe)
        )[0]

        # Get input/output stream info
        self.face_input_vstreams_params = InputVStreamParams.make(self.face_network_group)
        self.face_output_vstreams_params = OutputVStreamParams.make(self.face_network_group)

        self.landmark_input_vstreams_params = InputVStreamParams.make(self.landmark_network_group)
        self.landmark_output_vstreams_params = OutputVStreamParams.make(self.landmark_network_group)

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
                           self.face_input_vstreams_params,
                           self.face_output_vstreams_params) as infer_pipeline:
            # Get input stream name
            input_dict = {next(iter(infer_pipeline.input_vstreams)): input_data}
            output = infer_pipeline.infer(input_dict)

        # Parse face detections
        faces = self._parse_face_detections(output, image.shape)
        return faces

    def _parse_face_detections(self, output: dict, image_shape: tuple[int, int, int]) -> list[FaceBox]:
        """
        Parse face detection output

        Note: This is model-specific and may need adjustment based on your model's output format.
        The scrfd_2.5g model typically outputs bboxes in a specific format.
        """
        faces = []

        try:
            # Get output tensors
            output_keys = list(output.keys())
            print(f"DEBUG: Face detection output keys: {output_keys}")

            # SCRFD typically has multiple outputs
            # You may need to adjust this based on actual model output
            for key in output_keys:
                tensor = output[key]
                print(f"DEBUG: Output '{key}' shape: {tensor.shape}")

            # Example parsing (adjust based on your model's actual output)
            # This is a placeholder - you'll need to inspect the actual output format
            height, width = image_shape[:2]

            # TODO: Adjust this based on actual scrfd_2.5g output format
            # Common format: [batch, num_detections, 15] where 15 = bbox(4) + score(1) + landmarks(10)
            if len(output_keys) > 0:
                main_output = output[output_keys[0]]

                # Flatten and parse detections
                if len(main_output.shape) >= 2:
                    for detection in main_output[0]:  # Remove batch dimension
                        if len(detection) >= 5:
                            x1, y1, x2, y2, conf = detection[:5]

                            if conf > 0.5:  # Confidence threshold
                                faces.append(FaceBox(
                                    x=int(x1 * width),
                                    y=int(y1 * height),
                                    w=int((x2 - x1) * width),
                                    h=int((y2 - y1) * height),
                                    confidence=float(conf)
                                ))

            print(f"DEBUG: Detected {len(faces)} faces")

        except Exception as e:
            print(f"ERROR parsing face detections: {e}")
            print("You may need to adjust _parse_face_detections() for your model's output format")
            raise

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
        face_crop = image[face.y:face.y + face.h, face.x:face.x + face.w]

        # Resize to model input size (120x120 for tddfa_mobilenet_v1)
        input_size = 120
        resized = cv2.resize(face_crop, (input_size, input_size))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        with InferVStreams(self.landmark_network_group,
                           self.landmark_input_vstreams_params,
                           self.landmark_output_vstreams_params) as infer_pipeline:
            # Get input stream name
            input_dict = {next(iter(infer_pipeline.input_vstreams)): input_data}
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
        """
        Parse landmark detection output

        Note: tddfa_mobilenet_v1 outputs 68 landmarks in a specific format.
        You may need to adjust this based on actual model output.
        """
        try:
            # Get output tensor
            output_keys = list(output.keys())
            print(f"DEBUG: Landmark output keys: {output_keys}")

            for key in output_keys:
                tensor = output[key]
                print(f"DEBUG: Output '{key}' shape: {tensor.shape}")

            # Get the main output (adjust key if needed)
            output_key = output_keys[0]
            landmarks_raw = output[output_key]

            # Remove batch dimension if present
            if landmarks_raw.ndim > 2:
                landmarks_raw = landmarks_raw[0]

            print(f"DEBUG: Landmarks shape after processing: {landmarks_raw.shape}")

            # Reshape to (68, 2) if needed
            if landmarks_raw.size == 136:  # 68 points * 2 coordinates
                landmarks_raw = landmarks_raw.reshape(68, 2)
            elif landmarks_raw.shape == (68, 2):
                pass  # Already correct shape
            else:
                raise ValueError(
                    f"Unexpected landmark shape: {landmarks_raw.shape}. "
                    f"Expected (68, 2) or flattened (136,)"
                )

            # Scale landmarks back to original image coordinates
            h, w = crop_shape[:2]
            landmarks = landmarks_raw.copy()

            # Assuming landmarks are normalized [0, 1]
            landmarks[:, 0] = landmarks[:, 0] * w + face.x
            landmarks[:, 1] = landmarks[:, 1] * h + face.y

            return FaceLandmarks(
                points=landmarks,
                mouth_open=False,
                mouth_aspect_ratio=0.0
            )

        except Exception as e:
            print(f"ERROR parsing landmarks: {e}")
            print("You may need to adjust _parse_landmarks() for your model's output format")
            raise

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
        left_corner = landmarks[48]  # Left corner
        right_corner = landmarks[54]  # Right corner
        top_outer = landmarks[51]  # Top center outer
        bottom_outer = landmarks[57]  # Bottom center outer

        # Inner lip landmarks for better accuracy
        top_inner = landmarks[62]  # Top center inner
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
        # Network groups are automatically managed by VDevice
        # No explicit release needed
        print("Detector closed")


def main() -> None:
    """Main function to run face landmark detection"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hailo Face Landmark Detection with Mouth State Detection"
    )
    parser.add_argument(
        "--mouth-threshold",
        type=float,
        default=0.6,
        help="Threshold for mouth open detection (default: 0.6)"
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default="/usr/share/hailo-models/scrfd_2.5g.hef",
        help="Path to face detection HEF model"
    )
    parser.add_argument(
        "--landmark-model",
        type=str,
        default="/usr/share/hailo-models/tddfa_mobilenet_v1.hef",
        help="Path to landmark detection HEF model"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't auto-download models if missing"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Hailo Face Landmark Detection with Mouth State Detection")
    print("=" * 60)

    # Initialize detector (will auto-download models if not found)
    print("\n[1/3] Initializing detector...")
    try:
        detector = HailoFaceLandmarkDetector(
            face_detection_hef=args.face_model,
            landmark_hef=args.landmark_model,
            mouth_threshold=args.mouth_threshold
        )
    except FileNotFoundError as e:
        if args.no_download:
            print(f"\nERROR: {e}")
            print("Use --no-download=false to enable auto-download")
            return
        raise
    except Exception as e:
        print(f"\nERROR initializing detector: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Hailo AI HAT+ is properly connected")
        print("2. Check that hailo-all package is installed: sudo apt install hailo-all")
        print("3. Verify Hailo device: hailortcli fw-control identify")
        raise

    # Open camera
    print(f"\n[2/3] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open camera {args.camera}. "
            f"Try a different camera index with --camera N"
        )

    print("✓ Camera opened successfully")
    print(f"\n[3/3] Starting detection (mouth threshold: {args.mouth_threshold})...")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("=" * 60)

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Failed to read frame from camera")

            # Process frame
            annotated, landmarks = detector.process_frame(frame)

            # Display
            cv2.imshow("Hailo Face Landmarks - Mouth Detection", annotated)

            # Print mouth states (only every 30 frames to avoid spam)
            if frame_count % 30 == 0:
                if landmarks:
                    print(f"\nFrame {frame_count}:")
                    for i, lm in enumerate(landmarks):
                        status = "OPEN" if lm.mouth_open else "CLOSED"
                        print(f"  Face {i}: Mouth {status} (MAR: {lm.mouth_aspect_ratio:.3f})")
                else:
                    print(f"\nFrame {frame_count}: No faces detected")

            frame_count += 1

            # Check for quit or screenshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_path = f"hailo_face_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, annotated)
                print(f"✓ Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("\n✓ Cleanup complete")
        print("=" * 60)


if __name__ == "__main__":
    main()