"""
Async Target Tracker with Dual Detection: IMX500 Pose + Hailo Face Landmarks

ROS2-inspired architecture with simultaneous pose and face detection.
Hailo inference runs in dedicated worker thread to avoid blocking async loop.
"""
import asyncio
import logging
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from adafruit_servokit import ServoKit
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
from pydantic import BaseModel, Field

# Hailo imports
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# COCO Pose Keypoint Definitions
# ============================================================================

class CocoKeypoint(IntEnum):
    """COCO pose estimation keypoint indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


SKELETON_CONNECTIONS: list[tuple[CocoKeypoint, CocoKeypoint]] = [
    (CocoKeypoint.NOSE, CocoKeypoint.LEFT_EYE),
    (CocoKeypoint.NOSE, CocoKeypoint.RIGHT_EYE),
    (CocoKeypoint.LEFT_EYE, CocoKeypoint.LEFT_EAR),
    (CocoKeypoint.RIGHT_EYE, CocoKeypoint.RIGHT_EAR),
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.RIGHT_SHOULDER),
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.LEFT_HIP),
    (CocoKeypoint.RIGHT_SHOULDER, CocoKeypoint.RIGHT_HIP),
    (CocoKeypoint.LEFT_HIP, CocoKeypoint.RIGHT_HIP),
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.LEFT_ELBOW),
    (CocoKeypoint.LEFT_ELBOW, CocoKeypoint.LEFT_WRIST),
    (CocoKeypoint.RIGHT_SHOULDER, CocoKeypoint.RIGHT_ELBOW),
    (CocoKeypoint.RIGHT_ELBOW, CocoKeypoint.RIGHT_WRIST),
    (CocoKeypoint.LEFT_HIP, CocoKeypoint.LEFT_KNEE),
    (CocoKeypoint.LEFT_KNEE, CocoKeypoint.LEFT_ANKLE),
    (CocoKeypoint.RIGHT_HIP, CocoKeypoint.RIGHT_KNEE),
    (CocoKeypoint.RIGHT_KNEE, CocoKeypoint.RIGHT_ANKLE),
]


# ============================================================================
# Messages
# ============================================================================

@dataclass(frozen=True)
class FrameMessage:
    """Raw frame from camera."""
    frame: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class TargetLocationMessage:
    """Generic target location with orientation."""
    x: int | None
    y: int | None
    angle: float | None
    source: str  # "pose" or "face" or "brightness"
    timestamp: float


@dataclass(frozen=True)
class ServoStateMessage:
    """Motor node output."""
    pan_angle: float
    tilt_angle: float
    roll_angle: float
    is_locked_x: bool
    is_locked_y: bool
    is_locked_roll: bool
    timestamp: float


@dataclass(frozen=True)
class PoseDataMessage:
    """Full pose detection data for visualization."""
    keypoints: np.ndarray | None
    scores: np.ndarray | None
    boxes: list[np.ndarray] | None
    timestamp: float


@dataclass(frozen=True)
class FaceDataMessage:
    """Face detection and landmark data."""
    face_boxes: list[tuple[int, int, int, int]] | None  # [(x, y, w, h), ...]
    landmarks: list[np.ndarray] | None  # List of (68, 2) arrays
    mouth_states: list[bool] | None  # [is_open, ...]
    mouth_ratios: list[float] | None  # [MAR, ...]
    timestamp: float


# ============================================================================
# Pub/Sub System
# ============================================================================

class Topic:
    """Simple async topic with multiple subscribers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        """Create and return a new subscription queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._subscribers.append(queue)
        return queue

    async def publish(self, message: object) -> None:
        """Publish message to all subscribers (drop if queue full)."""
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass


class PubSub:
    """Pub/sub system."""

    def __init__(self) -> None:
        self.frame = Topic("frame")
        self.target_location = Topic("target_location")
        self.servo_state = Topic("servo_state")
        self.pose_data = Topic("pose_data")
        self.face_data = Topic("face_data")


# ============================================================================
# Node Parameters
# ============================================================================

class PoseDetectorParams(BaseModel):
    """Parameters for PoseDetectorNode."""
    model_path: str = Field(
        default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk",
        description="Path to IMX500 pose estimation model"
    )
    target_keypoint: CocoKeypoint = Field(
        default=CocoKeypoint.RIGHT_WRIST,
        description="Body part to track"
    )
    detection_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    keypoint_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    inference_rate: int = Field(default=10, ge=1, le=30)
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)


class HailoFaceDetectorParams(BaseModel):
    """Parameters for HailoFaceDetectorNode."""
    face_model_path: str = Field(
        default="/usr/share/hailo-models/scrfd_2.5g.hef",
        description="Path to face detection HEF"
    )
    landmark_model_path: str = Field(
        default="/usr/share/hailo-models/tddfa_mobilenet_v1.hef",
        description="Path to landmark detection HEF"
    )
    model_cache_dir: str = Field(
        default="~/.cache/hailo-models",
        description="Cache directory for auto-downloaded models"
    )
    mouth_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="MAR threshold for mouth open detection"
    )
    inference_rate: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Frames per second for face inference"
    )
    track_mouth: bool = Field(
        default=True,
        description="Track mouth position for servo control"
    )


class MotorNodeParams(BaseModel):
    """Parameters for MotorNode."""
    pan_channel: int = Field(default=11, ge=0, le=15)
    tilt_channel: int = Field(default=3, ge=0, le=15)
    roll_channel: int = Field(default=7, ge=0, le=15)
    target_x: int = Field(default=320, ge=0)
    target_y: int = Field(default=240, ge=0)
    gain: float = Field(default=0.05, gt=0.0, le=1.0)
    roll_gain: float = Field(default=0.8, gt=0.0, le=1.0)
    deadzone: int = Field(default=30, ge=0)
    roll_deadzone: float = Field(default=5.0, ge=0.0)
    tracking_source: Literal["pose", "face"] = Field(
        default="pose",
        description="Which detector to use for servo control"
    )


class UINodeParams(BaseModel):
    """Parameters for UINode."""
    deadzone: int = Field(default=30, ge=0)
    window_name: str = Field(default="Async Target Tracker")
    show_pose: bool = Field(default=True)
    show_face: bool = Field(default=True)


# ============================================================================
# Pose Detector Node (IMX500)
# ============================================================================

class PoseDetectorNode:
    """Detects human poses using IMX500."""

    def __init__(self, pubsub: PubSub, params: PoseDetectorParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        self._latest_keypoints: np.ndarray | None = None
        self._latest_scores: np.ndarray | None = None
        self._latest_boxes: list[np.ndarray] | None = None

        self.imx500 = IMX500(params.model_path)
        intrinsics = self.imx500.network_intrinsics

        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "pose estimation"
        elif intrinsics.task != "pose estimation":
            raise ValueError(f"Model is not a pose estimation task: {intrinsics.task}")

        intrinsics.inference_rate = params.inference_rate
        intrinsics.update_with_defaults()

        self.picam2 = Picamera2(self.imx500.camera_num)
        config = self.picam2.create_preview_configuration(
            controls={'FrameRate': params.inference_rate},
            buffer_count=12
        )
        self.picam2.configure(config)
        self.picam2.pre_callback = self._pose_callback

    def _pose_callback(self, request: CompletedRequest) -> None:
        """Process pose detection results."""
        metadata = request.get_metadata()
        np_outputs = self.imx500.get_outputs(metadata=metadata, add_batch=True)

        if np_outputs is None:
            return

        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=(self.params.height, self.params.width),
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=self.params.detection_threshold,
            network_postprocess=True
        )

        if scores is not None and len(scores) > 0:
            self._latest_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            self._latest_boxes = [np.array(b) for b in boxes]
            self._latest_scores = np.array(scores)
        else:
            self._latest_keypoints = None
            self._latest_scores = None
            self._latest_boxes = None

    def _calculate_body_angle(self, keypoints: np.ndarray) -> float | None:
        """Calculate body orientation from shoulder line."""
        left_shoulder = keypoints[CocoKeypoint.LEFT_SHOULDER]
        right_shoulder = keypoints[CocoKeypoint.RIGHT_SHOULDER]

        if left_shoulder[2] < self.params.keypoint_threshold or right_shoulder[2] < self.params.keypoint_threshold:
            return None

        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        angle = np.degrees(np.arctan2(dx, -dy))

        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180

        return float(angle)

    def _extract_target_keypoint(self) -> tuple[int, int, float] | None:
        """Extract target keypoint from latest detection results."""
        if self._latest_keypoints is None or self._latest_scores is None:
            return None

        best_person_idx = int(np.argmax(self._latest_scores))
        keypoints = self._latest_keypoints[best_person_idx]

        target_kp = keypoints[self.params.target_keypoint]

        if target_kp[2] < self.params.keypoint_threshold:
            return None

        x = int(target_kp[0])
        y = int(target_kp[1])
        angle = self._calculate_body_angle(keypoints)

        return (x, y, angle if angle is not None else 0.0)

    async def run(self) -> None:
        """Main detection loop."""
        logger.info(
            f"Starting PoseDetectorNode [target={self.params.target_keypoint.name}]"
        )

        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(show_preview=False)
        self.imx500.set_auto_aspect_ratio()

        await asyncio.sleep(1)
        self._running = True

        try:
            while self._running:
                result = self._extract_target_keypoint()

                if result:
                    x, y, angle = result
                else:
                    x, y, angle = None, None, None

                await self.pubsub.target_location.publish(
                    TargetLocationMessage(
                        x=x, y=y, angle=angle, source="pose", timestamp=time.time()
                    )
                )

                await self.pubsub.pose_data.publish(
                    PoseDataMessage(
                        keypoints=self._latest_keypoints,
                        scores=self._latest_scores,
                        boxes=self._latest_boxes,
                        timestamp=time.time()
                    )
                )

                frame = self.picam2.capture_array()
                await self.pubsub.frame.publish(
                    FrameMessage(frame=frame, timestamp=time.time())
                )

                await asyncio.sleep(0.01)
        finally:
            self.picam2.stop()
            logger.info("PoseDetectorNode stopped")

    async def stop(self) -> None:
        self._running = False


# ============================================================================
# Hailo Face Detector Node
# ============================================================================

class HailoFaceDetectorNode:
    """Detects faces and facial landmarks using Hailo AI HAT+.

    Runs in a dedicated thread to avoid blocking the async event loop.
    """

    MODEL_URLS = {
        "scrfd_2.5g": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l/scrfd_2.5g.hef",
        "tddfa_mobilenet_v1": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.0.0/tddfa_mobilenet_v1.hef",
    }

    def __init__(self, pubsub: PubSub, params: HailoFaceDetectorParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False
        self._thread = None
        self._event_loop = None  # Store reference to main async event loop

        self.frame_queue = pubsub.frame.subscribe()
        self.model_cache_dir = Path(params.model_cache_dir).expanduser()

        # Hailo resources will be initialized in worker thread
        self.device = None
        self.face_hef = None
        self.landmark_hef = None
        self.face_network_group = None
        self.landmark_network_group = None
        self.face_input_params = None
        self.face_output_params = None
        self.landmark_input_params = None
        self.landmark_output_params = None

        # Ensure models exist (can do this in main thread)
        logger.info("Checking face detection models...")
        self.face_model_path = self._ensure_model(params.face_model_path, "scrfd_2.5g")
        self.landmark_model_path = self._ensure_model(params.landmark_model_path, "tddfa_mobilenet_v1")

        self.frame_skip_counter = 0

    def _ensure_model(self, model_path: str, model_name: str) -> str:
        """Ensure model exists, download if necessary."""
        if Path(model_path).exists():
            logger.info(f"✓ Found {model_name} at: {model_path}")
            return model_path

        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.model_cache_dir / f"{model_name}.hef"

        if cache_path.exists():
            logger.info(f"✓ Found cached {model_name}")
            return str(cache_path)

        if model_name not in self.MODEL_URLS:
            raise FileNotFoundError(f"Model '{model_name}' not found")

        url = self.MODEL_URLS[model_name]
        logger.info(f"⬇ Downloading {model_name}...")

        try:
            urllib.request.urlretrieve(url, cache_path)
            logger.info(f"✓ Downloaded {model_name}")
            return str(cache_path)
        except Exception as e:
            if cache_path.exists():
                cache_path.unlink()
            raise RuntimeError(f"Failed to download {model_name}: {e}")

    def _detect_faces_simple(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Simplified face detection - returns list of (x, y, w, h)."""
        # For now, return a dummy face box covering center of frame
        # You'll need to implement actual SCRFD parsing based on model output
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        face_size = min(w, h) // 3

        return [(center_x - face_size//2, center_y - face_size//2, face_size, face_size)]

    def _initialize_hailo_resources(self) -> None:
        """Initialize Hailo resources (must be called from worker thread)."""
        logger.info("Initializing Hailo VDevice in worker thread...")
        try:
            self.device = VDevice()
            logger.info("✓ VDevice initialized")
        except Exception as e:
            logger.error(f"Failed to initialize VDevice: {e}")
            raise

        logger.info("Loading Hailo face models...")
        try:
            self.face_hef = HEF(self.face_model_path)
            logger.info("✓ Face detection HEF loaded")
            self.landmark_hef = HEF(self.landmark_model_path)
            logger.info("✓ Landmark detection HEF loaded")
        except Exception as e:
            logger.error(f"Failed to load HEF models: {e}")
            raise

        logger.info("Configuring network groups...")
        try:
            self.face_network_group = self.device.configure(
                self.face_hef,
                ConfigureParams.create_from_hef(self.face_hef, interface=HailoStreamInterface.PCIe)
            )[0]
            logger.info("✓ Face network group configured")

            self.landmark_network_group = self.device.configure(
                self.landmark_hef,
                ConfigureParams.create_from_hef(self.landmark_hef, interface=HailoStreamInterface.PCIe)
            )[0]
            logger.info("✓ Landmark network group configured")
        except Exception as e:
            logger.error(f"Failed to configure network groups: {e}")
            raise

        logger.info("Creating VStream parameters...")
        try:
            self.face_input_params = InputVStreamParams.make(self.face_network_group)
            self.face_output_params = OutputVStreamParams.make(self.face_network_group)

            self.landmark_input_params = InputVStreamParams.make(self.landmark_network_group)
            self.landmark_output_params = OutputVStreamParams.make(self.landmark_network_group)
            logger.info("✓ VStream parameters created")
        except Exception as e:
            logger.error(f"Failed to create VStream parameters: {e}")
            raise

    def _worker_thread(self) -> None:
        """Worker thread that runs synchronous Hailo operations."""
        logger.info(f"Face detector worker thread started (tid={threading.get_ident()})")

        try:
            # Initialize Hailo resources in this thread
            self._initialize_hailo_resources()

            frame_skip = max(1, 30 // self.params.inference_rate)

            while self._running:
                try:
                    # Blocking get from async queue (with timeout)
                    try:
                        frame_msg = self.frame_queue.get_nowait()
                    except:
                        time.sleep(0.01)
                        continue

                    # Skip frames to match inference rate
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter < frame_skip:
                        continue
                    self.frame_skip_counter = 0

                    # Detect faces (simplified)
                    face_boxes = self._detect_faces_simple(frame_msg.frame)

                    if not face_boxes:
                        # Publish empty data
                        asyncio.run_coroutine_threadsafe(
                            self.pubsub.face_data.publish(
                                FaceDataMessage(
                                    face_boxes=None,
                                    landmarks=None,
                                    mouth_states=None,
                                    mouth_ratios=None,
                                    timestamp=time.time()
                                )
                            ),
                            self._event_loop
                        )
                        asyncio.run_coroutine_threadsafe(
                            self.pubsub.target_location.publish(
                                TargetLocationMessage(
                                    x=None, y=None, angle=None, source="face", timestamp=time.time()
                                )
                            ),
                            self._event_loop
                        )
                        continue

                    # Process each face
                    all_landmarks = []
                    mouth_states = []
                    mouth_ratios = []

                    for face_box in face_boxes:
                        try:
                            landmarks = self._detect_landmarks_sync(frame_msg.frame, face_box)
                            all_landmarks.append(landmarks)

                            is_open, mar = self._calculate_mouth_state(landmarks)
                            mouth_states.append(is_open)
                            mouth_ratios.append(mar)
                        except Exception as e:
                            logger.error(f"Face processing error: {e}")
                            continue

                    # Publish face data
                    asyncio.run_coroutine_threadsafe(
                        self.pubsub.face_data.publish(
                            FaceDataMessage(
                                face_boxes=face_boxes,
                                landmarks=all_landmarks if all_landmarks else None,
                                mouth_states=mouth_states if mouth_states else None,
                                mouth_ratios=mouth_ratios if mouth_ratios else None,
                                timestamp=time.time()
                            )
                        ),
                        self._event_loop
                    )

                    # Publish target location
                    if self.params.track_mouth and len(all_landmarks) > 0:
                        landmarks = all_landmarks[0]
                        if landmarks.size > 0 and not np.all(landmarks == 0):
                            mouth_points = landmarks[48:60]
                            if mouth_points.size > 0:
                                mouth_center_x = int(np.mean(mouth_points[:, 0]))
                                mouth_center_y = int(np.mean(mouth_points[:, 1]))

                                asyncio.run_coroutine_threadsafe(
                                    self.pubsub.target_location.publish(
                                        TargetLocationMessage(
                                            x=mouth_center_x,
                                            y=mouth_center_y,
                                            angle=None,
                                            source="face",
                                            timestamp=time.time()
                                        )
                                    ),
                                    self._event_loop
                                )

                    time.sleep(0.01)

                except Exception as e:
                    logger.error(f"Worker thread error: {e}", exc_info=True)
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Fatal worker thread error: {e}", exc_info=True)
        finally:
            logger.info("Face detector worker thread stopped")
        """Detect 68 landmarks for a face (synchronous - run in executor)."""
        x, y, w, h = face_box

        # Crop face
        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            return np.zeros((68, 2))

        # Resize to 120x120
        resized = cv2.resize(face_crop, (120, 120))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        try:
            with InferVStreams(
                self.landmark_network_group,
                self.landmark_input_params,
                self.landmark_output_params
            ) as infer_pipeline:
                input_dict = {next(iter(infer_pipeline.input_vstreams)): input_data}
                output = infer_pipeline.infer(input_dict)

            # Parse landmarks (adjust based on actual model output)
            output_key = list(output.keys())[0]
            landmarks_raw = output[output_key]

            if landmarks_raw.ndim > 2:
                landmarks_raw = landmarks_raw[0]

            if landmarks_raw.size == 136:
                landmarks_raw = landmarks_raw.reshape(68, 2)

            # Scale back to original image coordinates
            landmarks = landmarks_raw.copy()
            landmarks[:, 0] = landmarks[:, 0] * w + x
            landmarks[:, 1] = landmarks[:, 1] * h + y

            return landmarks

        except Exception as e:
            logger.warning(f"Landmark detection failed: {e}")
            return np.zeros((68, 2))

    def _calculate_mouth_state(self, landmarks: np.ndarray) -> tuple[bool, float]:
        """Calculate mouth aspect ratio (MAR) for open/closed detection."""
        if landmarks.size == 0:
            return False, 0.0

        # Mouth landmarks (iBUG 68-point format: 48-67)
        try:
            left_corner = landmarks[48]
            right_corner = landmarks[54]
            top_outer = landmarks[51]
            bottom_outer = landmarks[57]
            top_inner = landmarks[62]
            bottom_inner = landmarks[66]

            width = np.linalg.norm(right_corner - left_corner)
            height_outer = np.linalg.norm(top_outer - bottom_outer)
            height_inner = np.linalg.norm(top_inner - bottom_inner)
            height = (height_outer + height_inner) / 2.0

            mar = height / (width + 1e-6)
            is_open = mar > self.params.mouth_threshold

            return is_open, float(mar)
        except:
            return False, 0.0

    async def run(self) -> None:
        """Main face detection loop - launches worker thread."""
        logger.info(f"Starting HailoFaceDetectorNode [rate={self.params.inference_rate}fps]")
        self._running = True

        # Store reference to main event loop for worker thread
        self._event_loop = asyncio.get_running_loop()

        # Start worker thread for blocking Hailo operations
        self._thread = threading.Thread(target=self._worker_thread, daemon=True)
        self._thread.start()

        try:
            # Just keep the async task alive while worker thread runs
            while self._running and self._thread.is_alive():
                await asyncio.sleep(0.5)

            if not self._thread.is_alive():
                logger.error("Face detector worker thread died unexpectedly")

        except Exception as e:
            logger.error(f"HailoFaceDetectorNode error: {e}", exc_info=True)
        finally:
            logger.info("HailoFaceDetectorNode stopped")

    async def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


# ============================================================================
# Motor Node
# ============================================================================

class MotorNode:
    """Generic servo controller."""

    def __init__(self, pubsub: PubSub, params: MotorNodeParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        self.kit = ServoKit(channels=16)
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        self.roll_angle = 90.0
        self.kit.servo[params.pan_channel].angle = self.pan_angle
        self.kit.servo[params.tilt_channel].angle = self.tilt_angle
        self.kit.servo[params.roll_channel].angle = self.roll_angle

        self.target_queue = pubsub.target_location.subscribe()

    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(f"Starting MotorNode [tracking={self.params.tracking_source}]")
        self._running = True

        try:
            while self._running:
                msg: TargetLocationMessage = await self.target_queue.get()

                # Only respond to messages from the selected source
                if msg.source != self.params.tracking_source:
                    continue

                if msg.x is None or msg.y is None:
                    await self.pubsub.servo_state.publish(
                        ServoStateMessage(
                            pan_angle=self.pan_angle,
                            tilt_angle=self.tilt_angle,
                            roll_angle=self.roll_angle,
                            is_locked_x=False,
                            is_locked_y=False,
                            is_locked_roll=False,
                            timestamp=time.time()
                        )
                    )
                    continue

                error_x = msg.x - self.params.target_x
                error_y = msg.y - self.params.target_y

                is_locked_x = abs(error_x) <= self.params.deadzone
                is_locked_y = abs(error_y) <= self.params.deadzone

                if not is_locked_x:
                    self.pan_angle += error_x * self.params.gain
                    self.pan_angle = float(np.clip(self.pan_angle, 0.0, 180.0))
                    self.kit.servo[self.params.pan_channel].angle = self.pan_angle

                if not is_locked_y:
                    self.tilt_angle += error_y * self.params.gain
                    self.tilt_angle = float(np.clip(self.tilt_angle, 0.0, 180.0))
                    self.kit.servo[self.params.tilt_channel].angle = self.tilt_angle

                is_locked_roll = False
                if msg.angle is not None:
                    target_roll = 90.0 + msg.angle
                    roll_error = target_roll - self.roll_angle
                    is_locked_roll = abs(roll_error) <= self.params.roll_deadzone

                    if not is_locked_roll:
                        self.roll_angle += roll_error * self.params.roll_gain
                        self.roll_angle = float(np.clip(self.roll_angle, 0.0, 180.0))
                        self.kit.servo[self.params.roll_channel].angle = self.roll_angle

                await self.pubsub.servo_state.publish(
                    ServoStateMessage(
                        pan_angle=self.pan_angle,
                        tilt_angle=self.tilt_angle,
                        roll_angle=self.roll_angle,
                        is_locked_x=is_locked_x,
                        is_locked_y=is_locked_y,
                        is_locked_roll=is_locked_roll,
                        timestamp=time.time()
                    )
                )
        finally:
            self.kit.servo[self.params.pan_channel].angle = 90.0
            self.kit.servo[self.params.tilt_channel].angle = 90.0
            self.kit.servo[self.params.roll_channel].angle = 90.0
            await asyncio.sleep(0.5)
            self.kit.servo[self.params.pan_channel].angle = None
            self.kit.servo[self.params.tilt_channel].angle = None
            self.kit.servo[self.params.roll_channel].angle = None
            logger.info("MotorNode stopped")

    async def stop(self) -> None:
        self._running = False


# ============================================================================
# UI Node
# ============================================================================

class UINode:
    """Displays camera feed with both pose and face tracking visualization."""

    def __init__(self, pubsub: PubSub, params: UINodeParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        self.frame_queue = pubsub.frame.subscribe()
        self.target_queue = pubsub.target_location.subscribe()
        self.servo_state_queue = pubsub.servo_state.subscribe()
        self.pose_data_queue = pubsub.pose_data.subscribe()
        self.face_data_queue = pubsub.face_data.subscribe()

        self.latest_targets: dict[str, TargetLocationMessage] = {}
        self.latest_servo_state: ServoStateMessage | None = None
        self.latest_pose_data: PoseDataMessage | None = None
        self.latest_face_data: FaceDataMessage | None = None

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.3) -> None:
        """Draw pose skeleton."""
        for kp1, kp2 in SKELETON_CONNECTIONS:
            pt1 = keypoints[kp1]
            pt2 = keypoints[kp2]

            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for i, kp in enumerate(keypoints):
            if kp[2] > confidence_threshold:
                x, y = int(kp[0]), int(kp[1])
                if i <= 4:
                    color = (255, 0, 0)
                elif i <= 10:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

    def _draw_face_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, is_mouth_open: bool) -> None:
        """Draw 68 facial landmarks."""
        mouth_color = (0, 0, 255) if is_mouth_open else (255, 0, 0)

        for i, (x, y) in enumerate(landmarks.astype(int)):
            if 48 <= i < 68:  # Mouth region
                color = mouth_color
                radius = 3
            else:
                color = (0, 255, 255)
                radius = 2
            cv2.circle(frame, (x, y), radius, color, -1)

    def _draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Draw all tracking visualizations."""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        # Draw pose skeletons
        if self.params.show_pose and self.latest_pose_data and self.latest_pose_data.keypoints is not None:
            for person_keypoints in self.latest_pose_data.keypoints:
                self._draw_skeleton(frame, person_keypoints)

        # Draw face landmarks
        if self.params.show_face and self.latest_face_data:
            if self.latest_face_data.landmarks:
                for i, landmarks in enumerate(self.latest_face_data.landmarks):
                    is_open = self.latest_face_data.mouth_states[i] if self.latest_face_data.mouth_states else False
                    self._draw_face_landmarks(frame, landmarks, is_open)

            # Draw face boxes
            if self.latest_face_data.face_boxes:
                for x, y, w, h in self.latest_face_data.face_boxes:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Center crosshair
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        cv2.line(frame, (0, center_y), (w, center_y), (255, 255, 255), 2)

        # Deadzone box
        cv2.rectangle(
            frame,
            (center_x - self.params.deadzone, center_y - self.params.deadzone),
            (center_x + self.params.deadzone, center_y + self.params.deadzone),
            (128, 128, 128), 1
        )

        # Draw targets from both sources
        for source, target_msg in self.latest_targets.items():
            if target_msg.x is not None:
                x, y = target_msg.x, target_msg.y

                # Different colors for different sources
                if source == "pose":
                    color = (0, 255, 0) if self.latest_servo_state and self.latest_servo_state.is_locked_x else (255, 0, 0)
                    label = "POSE"
                else:
                    color = (255, 255, 0)
                    label = "FACE"

                cv2.circle(frame, (x, y), 15, color, 2)
                cv2.putText(frame, label, (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status overlay
        y_offset = 30
        if self.latest_servo_state:
            status = "LOCKED" if (
                self.latest_servo_state.is_locked_x and
                self.latest_servo_state.is_locked_y
            ) else "TRACKING"
            status_color = (0, 255, 0) if status == "LOCKED" else (255, 255, 0)

            cv2.putText(frame, status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            y_offset += 30
            cv2.putText(frame, f"Pan: {self.latest_servo_state.pan_angle:.1f}°", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(frame, f"Tilt: {self.latest_servo_state.tilt_angle:.1f}°", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25

        # Face detection info
        if self.latest_face_data and self.latest_face_data.mouth_states:
            for i, (is_open, mar) in enumerate(zip(self.latest_face_data.mouth_states, self.latest_face_data.mouth_ratios)):
                status = "OPEN" if is_open else "CLOSED"
                cv2.putText(frame, f"Mouth {i}: {status} ({mar:.2f})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    async def _update_state(self) -> None:
        """Background task to consume queue updates."""
        while self._running:
            try:
                while not self.target_queue.empty():
                    msg = self.target_queue.get_nowait()
                    self.latest_targets[msg.source] = msg

                while not self.servo_state_queue.empty():
                    self.latest_servo_state = self.servo_state_queue.get_nowait()

                while not self.pose_data_queue.empty():
                    self.latest_pose_data = self.pose_data_queue.get_nowait()

                while not self.face_data_queue.empty():
                    self.latest_face_data = self.face_data_queue.get_nowait()

                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break

    async def run(self) -> None:
        """Main UI loop."""
        logger.info(f"Starting UINode")
        self._running = True

        update_task = asyncio.create_task(self._update_state())

        try:
            while self._running:
                frame_msg: FrameMessage = await self.frame_queue.get()
                vis_frame = self._draw_visualization(frame_msg.frame)

                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = time.time()

                cv2.imshow(self.params.window_name, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                await asyncio.sleep(0.001)
        finally:
            update_task.cancel()
            cv2.destroyAllWindows()
            logger.info("UINode stopped")

    async def stop(self) -> None:
        self._running = False


# ============================================================================
# Launch System
# ============================================================================

class LaunchConfig(BaseModel):
    """Declarative launch configuration."""
    pose_detector: PoseDetectorParams = Field(default_factory=PoseDetectorParams)
    face_detector: HailoFaceDetectorParams = Field(default_factory=HailoFaceDetectorParams)
    motor: MotorNodeParams = Field(default_factory=MotorNodeParams)
    ui: UINodeParams = Field(default_factory=UINodeParams)
    enable_face_detection: bool = Field(default=True, description="Enable Hailo face detection")


class Launcher:
    """Launches nodes based on declarative config."""

    def __init__(self, config: LaunchConfig) -> None:
        self.config = config
        self.pubsub = PubSub()
        self.nodes: list = []

    def _create_nodes(self) -> None:
        """Instantiate nodes based on config."""
        logger.info("Creating nodes from launch config...")

        # Always create pose detector
        self.nodes.append(PoseDetectorNode(self.pubsub, self.config.pose_detector))

        # Optionally create face detector
        if self.config.enable_face_detection:
            self.nodes.append(HailoFaceDetectorNode(self.pubsub, self.config.face_detector))

        # Always create motor and UI
        self.nodes.append(MotorNode(self.pubsub, self.config.motor))
        self.nodes.append(UINode(self.pubsub, self.config.ui))

        logger.info(f"Created {len(self.nodes)} nodes")

    async def run(self) -> None:
        """Launch all nodes."""
        self._create_nodes()

        logger.info("=" * 60)
        logger.info("LAUNCHING DUAL TRACKER (IMX500 + HAILO)")
        logger.info("=" * 60)

        try:
            await asyncio.gather(*[node.run() for node in self.nodes])
        except KeyboardInterrupt:
            logger.info("\nStopping...")
        finally:
            for node in self.nodes:
                await node.stop()
            logger.info("All nodes stopped")


# ============================================================================
# Main
# ============================================================================

async def main() -> None:
    """Launch with both pose and face detection."""

    # OPTION 1: Both detectors (may need debugging)
    config = LaunchConfig(
        enable_face_detection=True,
        pose_detector=PoseDetectorParams(
            target_keypoint=CocoKeypoint.RIGHT_WRIST,
            detection_threshold=0.3,
            keypoint_threshold=0.3,
        ),
        face_detector=HailoFaceDetectorParams(
            mouth_threshold=0.6,
            inference_rate=5,
            track_mouth=True,
        ),
        motor=MotorNodeParams(
            tracking_source="pose",  # Change to "face" to track mouth
            gain=0.05,
            deadzone=30,
        ),
        ui=UINodeParams(
            show_pose=True,
            show_face=True,
        ),
    )

    # OPTION 2: Just pose tracking (if face detection has issues)
    # config = LaunchConfig(
    #     enable_face_detection=False,
    #     pose_detector=PoseDetectorParams(
    #         target_keypoint=CocoKeypoint.RIGHT_WRIST,
    #     ),
    #     motor=MotorNodeParams(
    #         tracking_source="pose",
    #         gain=0.05,
    #         deadzone=30,
    #     ),
    #     ui=UINodeParams(
    #         show_pose=True,
    #         show_face=False,
    #     ),
    # )

    launcher = Launcher(config)
    await launcher.run()


if __name__ == "__main__":
    asyncio.run(main())