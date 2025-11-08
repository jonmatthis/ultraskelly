import asyncio
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from pydantic import BaseModel, Field

from ultraskelly.core.bot.__main__bot import logger
from ultraskelly.core.bot.sensory.camera_node import FrameMessage
from ultraskelly.core.bot.motor.head_node import TargetLocationMessage
from ultraskelly.core.bot.pubsub import PubSub


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
    # Head
    (CocoKeypoint.NOSE, CocoKeypoint.LEFT_EYE),
    (CocoKeypoint.NOSE, CocoKeypoint.RIGHT_EYE),
    (CocoKeypoint.LEFT_EYE, CocoKeypoint.LEFT_EAR),
    (CocoKeypoint.RIGHT_EYE, CocoKeypoint.RIGHT_EAR),
    # Torso
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.RIGHT_SHOULDER),
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.LEFT_HIP),
    (CocoKeypoint.RIGHT_SHOULDER, CocoKeypoint.RIGHT_HIP),
    (CocoKeypoint.LEFT_HIP, CocoKeypoint.RIGHT_HIP),
    # Left arm
    (CocoKeypoint.LEFT_SHOULDER, CocoKeypoint.LEFT_ELBOW),
    (CocoKeypoint.LEFT_ELBOW, CocoKeypoint.LEFT_WRIST),
    # Right arm
    (CocoKeypoint.RIGHT_SHOULDER, CocoKeypoint.RIGHT_ELBOW),
    (CocoKeypoint.RIGHT_ELBOW, CocoKeypoint.RIGHT_WRIST),
    # Left leg
    (CocoKeypoint.LEFT_HIP, CocoKeypoint.LEFT_KNEE),
    (CocoKeypoint.LEFT_KNEE, CocoKeypoint.LEFT_ANKLE),
    # Right leg
    (CocoKeypoint.RIGHT_HIP, CocoKeypoint.RIGHT_KNEE),
    (CocoKeypoint.RIGHT_KNEE, CocoKeypoint.RIGHT_ANKLE),
]


@dataclass(frozen=True)
class PoseDataMessage:
    """Full pose detection data for visualization."""
    keypoints: np.ndarray | None  # Shape: (num_people, 17, 3) where 3 = [x, y, confidence]
    scores: np.ndarray | None  # Shape: (num_people,)
    boxes: list[np.ndarray] | None
    timestamp: float


class PoseDetectorParams(BaseModel):
    """Parameters for PoseDetectorNode."""
    model_path: str = Field(
        default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk",
        description="Path to IMX500 pose estimation model"
    )
    target_keypoint: CocoKeypoint = Field(
        default=CocoKeypoint.NOSE,
        description="Body part to track"
    )
    detection_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for person detection"
    )
    keypoint_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for keypoint detection"
    )
    inference_rate: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Frames per second for inference"
    )
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)


class PoseDetectorNode:
    """Detects human poses using IMX500 and tracks specific body part."""

    def __init__(self, pubsub: PubSub, params: PoseDetectorParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        # Latest detection results (shared between callback and async task)
        self._latest_keypoints: np.ndarray | None = None
        self._latest_scores: np.ndarray | None = None
        self._latest_boxes: list[np.ndarray] | None = None
        self._detection_lock = asyncio.Lock()

        # Initialize IMX500 before Picamera2
        self.imx500 = IMX500(params.model_path)
        intrinsics = self.imx500.network_intrinsics

        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "pose estimation"
        elif intrinsics.task != "pose estimation":
            raise ValueError(f"Model is not a pose estimation task: {intrinsics.task}")

        intrinsics.inference_rate = params.inference_rate
        intrinsics.update_with_defaults()

        # Initialize camera
        self.picam2 = Picamera2(self.imx500.camera_num)
        config = self.picam2.create_preview_configuration(
            controls={'FrameRate': params.inference_rate},
            buffer_count=12
        )
        self.picam2.configure(config)
        self.picam2.pre_callback = self._pose_callback

    def _pose_callback(self, request: CompletedRequest) -> None:
        """Callback to process pose detection results (runs in picamera2 thread)."""
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

        # Store results (thread-safe update)
        if scores is not None and len(scores) > 0:
            # Reshape keypoints to (num_people, 17, 3) where 3 = [x, y, confidence]
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

        # Check confidence
        if left_shoulder[2] < self.params.keypoint_threshold or right_shoulder[2] < self.params.keypoint_threshold:
            return None

        # Calculate angle of shoulder line
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]

        # Convert to degrees, 0° = vertical
        angle = np.degrees(np.arctan2(dx, -dy))

        # Normalize to [-90, 90]
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180

        return float(angle)

    def _extract_target_keypoint(self) -> tuple[int, int, float] | None:
        """Extract target keypoint from latest detection results."""
        if self._latest_keypoints is None or self._latest_scores is None:
            return None

        # Find person with highest score
        best_person_idx = int(np.argmax(self._latest_scores))
        keypoints = self._latest_keypoints[best_person_idx]

        # Get target keypoint (enum value IS the index)
        target_kp = keypoints[self.params.target_keypoint]

        # Check confidence
        if target_kp[2] < self.params.keypoint_threshold:
            return None

        x = int(target_kp[0])
        y = int(target_kp[1])

        # Calculate body orientation
        angle = self._calculate_body_angle(keypoints)

        return (x, y, angle if angle is not None else 0.0)

    async def run(self) -> None:
        """Main detection loop."""
        logger.info(
            f"Starting PoseDetectorNode [target={self.params.target_keypoint.name}, "
            f"threshold={self.params.detection_threshold}]"
        )

        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(show_preview=False)
        self.imx500.set_auto_aspect_ratio()

        await asyncio.sleep(1)
        self._running = True

        try:
            while self._running:
                # Extract target from latest detection
                result = self._extract_target_keypoint()

                if result:
                    x, y, angle = result
                else:
                    x, y, angle = None, None, None

                await self.pubsub.target_location.publish(
                    TargetLocationMessage(x=x, y=y, angle=angle, timestamp=time.time())
                )

                # Publish full pose data for visualization
                await self.pubsub.pose_data.publish(
                    PoseDataMessage(
                        keypoints=self._latest_keypoints,
                        scores=self._latest_scores,
                        boxes=self._latest_boxes,
                        timestamp=time.time()
                    )
                )

                # Publish frames for UI
                frame = self.picam2.capture_array()
                await self.pubsub.frame.publish(
                    FrameMessage(frame=frame, timestamp=time.time())
                )

                await asyncio.sleep(0.01)  # 100 Hz publishing rate
        finally:
            self.picam2.stop()
            logger.info("PoseDetectorNode stopped")

    async def stop(self) -> None:
        self._running = False
