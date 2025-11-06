"""
Async Target Tracker with Pose Estimation Integration

ROS2-inspired architecture: independent nodes + parameter system + launch config.
Now includes IMX500-based human pose tracking.
"""
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import cv2
import numpy as np
from adafruit_servokit import ServoKit
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
from pydantic import BaseModel, Field

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


# Skeleton connections (bones) for visualization
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
    angle: float | None  # Rotation angle in degrees (0° = vertical up)
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
    keypoints: np.ndarray | None  # Shape: (num_people, 17, 3) where 3 = [x, y, confidence]
    scores: np.ndarray | None  # Shape: (num_people,)
    boxes: list[np.ndarray] | None
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


# ============================================================================
# Node Parameters (ROS2-inspired)
# ============================================================================

class VisionNodeParams(BaseModel):
    """Parameters for VisionNode."""
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)


class BrightnessDetectorParams(BaseModel):
    """Parameters for BrightnessDetectorNode."""
    blur_size: int = Field(default=15, ge=1, description="Must be odd")
    threshold: int = Field(default=100, ge=0, le=255)


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


class MotorNodeParams(BaseModel):
    """Parameters for MotorNode."""
    pan_channel: int = Field(default=11, ge=0, le=15)
    tilt_channel: int = Field(default=3, ge=0, le=15)
    roll_channel: int = Field(default=7, ge=0, le=15)
    target_x: int = Field(default=320, ge=0)
    target_y: int = Field(default=240, ge=0)
    gain: float = Field(default=0.05, gt=0.0, le=1.0)
    roll_gain: float = Field(default=0.8, gt=0.0, le=1.0, description="How aggressively to match roll angle")
    deadzone: int = Field(default=30, ge=0)
    roll_deadzone: float = Field(default=5.0, ge=0.0, description="Roll angle deadzone in degrees")


class UINodeParams(BaseModel):
    """Parameters for UINode."""
    deadzone: int = Field(default=30, ge=0)
    window_name: str = Field(default="Async Target Tracker")


# ============================================================================
# Vision Node (for brightness detector)
# ============================================================================

class VisionNode:
    """Captures frames and publishes them."""

    def __init__(self, pubsub: PubSub, params: VisionNodeParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={'size': (params.width, params.height), 'format': 'RGB888'}
        )
        self.picam2.configure(config)

    async def run(self) -> None:
        """Main vision node loop."""
        logger.info(f"Starting VisionNode [{self.params.width}x{self.params.height}]")
        self.picam2.start()
        await asyncio.sleep(1)
        self._running = True

        try:
            while self._running:
                frame = self.picam2.capture_array()
                await self.pubsub.frame.publish(
                    FrameMessage(frame=frame, timestamp=time.time())
                )
                await asyncio.sleep(0.001)
        finally:
            self.picam2.stop()
            logger.info("VisionNode stopped")

    async def stop(self) -> None:
        self._running = False


# ============================================================================
# Detector Nodes
# ============================================================================

class BrightnessDetectorNode:
    """Detects brightest point in frame with orientation."""

    def __init__(self, pubsub: PubSub, params: BrightnessDetectorParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False
        self.frame_queue = pubsub.frame.subscribe()

    def _detect_target(self, frame: np.ndarray) -> tuple[int, int, float] | None:
        """Find brightest point and its orientation in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.params.blur_size, self.params.blur_size), 0)

        # Threshold to get bright region
        _, binary = cv2.threshold(blurred, self.params.threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Need at least 5 points to fit ellipse
        if len(largest_contour) < 5:
            # Fall back to centroid only
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                return None
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy, 0.0)  # No rotation if we can't determine it

        # Fit ellipse to get orientation
        ellipse = cv2.fitEllipse(largest_contour)
        center, axes, angle = ellipse

        # Convert angle: OpenCV gives angle where 0° is horizontal right
        # We want 0° to be vertical up, so rotate by -90°
        rotation_angle = angle - 90.0

        # Normalize to [-90, 90] range
        while rotation_angle > 90:
            rotation_angle -= 180
        while rotation_angle < -90:
            rotation_angle += 180

        return (int(center[0]), int(center[1]), rotation_angle)

    async def run(self) -> None:
        """Main detection loop."""
        logger.info(
            f"Starting BrightnessDetectorNode [blur={self.params.blur_size}, threshold={self.params.threshold}]")
        self._running = True

        try:
            while self._running:
                frame_msg: FrameMessage = await self.frame_queue.get()
                result = self._detect_target(frame_msg.frame)

                if result:
                    x, y, angle = result
                else:
                    x, y, angle = None, None, None

                await self.pubsub.target_location.publish(
                    TargetLocationMessage(x=x, y=y, angle=angle, timestamp=time.time())
                )
        finally:
            logger.info("BrightnessDetectorNode stopped")

    async def stop(self) -> None:
        self._running = False


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


# ============================================================================
# Motor Node
# ============================================================================

class MotorNode:
    """Generic servo controller - tracks whatever target is published."""

    def __init__(self, pubsub: PubSub, params: MotorNodeParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        # Initialize servos
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
        logger.info(f"Starting MotorNode [gain={self.params.gain}, deadzone={self.params.deadzone}px, roll_gain={self.params.roll_gain}]")
        self._running = True

        try:
            while self._running:
                msg: TargetLocationMessage = await self.target_queue.get()

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

                # Calculate errors
                error_x = msg.x - self.params.target_x
                error_y = msg.y - self.params.target_y

                # Check lock status
                is_locked_x = abs(error_x) <= self.params.deadzone
                is_locked_y = abs(error_y) <= self.params.deadzone

                # Update pan/tilt servos if not locked
                if not is_locked_x:
                    self.pan_angle += error_x * self.params.gain
                    self.pan_angle = float(np.clip(self.pan_angle, 0.0, 180.0))
                    self.kit.servo[self.params.pan_channel].angle = self.pan_angle

                if not is_locked_y:
                    self.tilt_angle += error_y * self.params.gain
                    self.tilt_angle = float(np.clip(self.tilt_angle, 0.0, 180.0))
                    self.kit.servo[self.params.tilt_channel].angle = self.tilt_angle

                # Handle roll angle if detected
                is_locked_roll = False
                if msg.angle is not None:
                    # Map rotation angle [-90, 90] to servo angle [0, 180]
                    target_roll = 90.0 + msg.angle

                    # Calculate roll error
                    roll_error = target_roll - self.roll_angle

                    # Check if roll is locked
                    is_locked_roll = abs(roll_error) <= self.params.roll_deadzone

                    # Update roll servo if not locked
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
            # Center servos
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
    """Displays camera feed with tracking visualization."""

    def __init__(self, pubsub: PubSub, params: UINodeParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        self.frame_queue = pubsub.frame.subscribe()
        self.target_queue = pubsub.target_location.subscribe()
        self.servo_state_queue = pubsub.servo_state.subscribe()
        self.pose_data_queue = pubsub.pose_data.subscribe()

        self.latest_target: TargetLocationMessage | None = None
        self.latest_servo_state: ServoStateMessage | None = None
        self.latest_pose_data: PoseDataMessage | None = None

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.3) -> None:
        """Draw skeleton on frame for a single person."""
        # Draw bones
        for kp1, kp2 in SKELETON_CONNECTIONS:
            pt1 = keypoints[kp1]
            pt2 = keypoints[kp2]

            # Only draw if both keypoints are confident
            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw keypoints on top of bones
        for i, kp in enumerate(keypoints):
            if kp[2] > confidence_threshold:
                x, y = int(kp[0]), int(kp[1])
                # Different color for different body parts
                if i <= 4:  # Head
                    color = (255, 0, 0)  # Blue
                elif i <= 10:  # Arms
                    color = (0, 255, 0)  # Green
                else:  # Legs
                    color = (0, 0, 255)  # Red
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

    def _draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking visualization on frame."""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        # Draw all detected skeletons
        if self.latest_pose_data and self.latest_pose_data.keypoints is not None:
            for person_keypoints in self.latest_pose_data.keypoints:
                self._draw_skeleton(frame, person_keypoints, confidence_threshold=0.3)

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

        # Target point
        if self.latest_target and self.latest_target.x is not None:
            x = self.latest_target.x
            y = self.latest_target.y

            is_locked = (
                self.latest_servo_state.is_locked_x and
                self.latest_servo_state.is_locked_y and
                self.latest_servo_state.is_locked_roll
                if self.latest_servo_state else False
            )

            color = (0, 255, 0) if is_locked else (255, 0, 0)
            cv2.circle(frame, (x, y), 20, color, 3)

            # Draw orientation line if angle is available
            if self.latest_target.angle is not None:
                angle_rad = np.radians(self.latest_target.angle)
                line_length = 40
                end_x = int(x + line_length * np.sin(angle_rad))
                end_y = int(y - line_length * np.cos(angle_rad))

                roll_color = (0, 255, 0) if (self.latest_servo_state and self.latest_servo_state.is_locked_roll) else (255, 0, 255)
                cv2.line(frame, (x, y), (end_x, end_y), roll_color, 3)

            if self.latest_servo_state:
                line_x_color = (0, 255, 0) if self.latest_servo_state.is_locked_x else (255, 255, 0)
                line_y_color = (0, 255, 0) if self.latest_servo_state.is_locked_y else (255, 255, 0)
                cv2.line(frame, (center_x, y), (x, y), line_x_color, 2)
                cv2.line(frame, (x, center_y), (x, y), line_y_color, 2)

        # Status overlay
        if self.latest_servo_state:
            status = "LOCKED" if (
                    self.latest_servo_state.is_locked_x and
                    self.latest_servo_state.is_locked_y and
                    self.latest_servo_state.is_locked_roll
            ) else "TRACKING"
            status_color = (0, 255, 0) if status == "LOCKED" else (255, 255, 0)

            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(
                frame,
                f"Pan: {self.latest_servo_state.pan_angle:.1f}°",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            cv2.putText(
                frame,
                f"Tilt: {self.latest_servo_state.tilt_angle:.1f}°",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            cv2.putText(
                frame,
                f"Roll: {self.latest_servo_state.roll_angle:.1f}°",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        # Target angle display
        if self.latest_target and self.latest_target.angle is not None:
            cv2.putText(
                frame,
                f"Body Angle: {self.latest_target.angle:.1f}°",
                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1
            )

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    async def _update_state(self) -> None:
        """Background task to consume queue updates."""
        while self._running:
            try:
                while not self.target_queue.empty():
                    self.latest_target = self.target_queue.get_nowait()

                while not self.servo_state_queue.empty():
                    self.latest_servo_state = self.servo_state_queue.get_nowait()

                while not self.pose_data_queue.empty():
                    self.latest_pose_data = self.pose_data_queue.get_nowait()

                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break

    async def run(self) -> None:
        """Main UI loop."""
        logger.info(f"Starting UINode [window='{self.params.window_name}']")
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
# Launch System (ROS2-inspired)
# ============================================================================

class LaunchConfig(BaseModel):
    """Declarative launch configuration."""

    detector_type: Literal["brightness", "pose"] = Field(
        default="pose",
        description="Which detector to use"
    )

    vision: VisionNodeParams = Field(default_factory=VisionNodeParams)
    brightness_detector: BrightnessDetectorParams = Field(default_factory=BrightnessDetectorParams)
    pose_detector: PoseDetectorParams = Field(default_factory=PoseDetectorParams)
    motor: MotorNodeParams = Field(default_factory=MotorNodeParams)
    ui: UINodeParams = Field(default_factory=UINodeParams)


class Launcher:
    """Launches nodes based on declarative config."""

    def __init__(self, config: LaunchConfig) -> None:
        self.config = config
        self.pubsub = PubSub()
        self.nodes: list[VisionNode | BrightnessDetectorNode | PoseDetectorNode | MotorNode | UINode] = []

    def _create_nodes(self) -> None:
        """Instantiate nodes based on config."""
        logger.info("Creating nodes from launch config...")

        # Always create motor and UI
        self.nodes.append(MotorNode(self.pubsub, self.config.motor))
        self.nodes.append(UINode(self.pubsub, self.config.ui))

        # Create detector based on type
        if self.config.detector_type == "brightness":
            self.nodes.append(VisionNode(self.pubsub, self.config.vision))
            self.nodes.append(
                BrightnessDetectorNode(self.pubsub, self.config.brightness_detector)
            )
        elif self.config.detector_type == "pose":
            self.nodes.append(
                PoseDetectorNode(self.pubsub, self.config.pose_detector)
            )
        else:
            raise ValueError(f"Unknown detector type: {self.config.detector_type}")

        logger.info(f"Created {len(self.nodes)} nodes")

    async def run(self) -> None:
        """Launch all nodes."""
        self._create_nodes()

        logger.info("=" * 60)
        logger.info("LAUNCHING ASYNC TARGET TRACKER")
        logger.info("=" * 60)

        try:
            # Run all nodes concurrently
            await asyncio.gather(*[node.run() for node in self.nodes])
        except KeyboardInterrupt:
            logger.info("\nStopping...")
        finally:
            # Stop all nodes
            for node in self.nodes:
                await node.stop()
            logger.info("All nodes stopped")


# ============================================================================
# Main - Declarative Configuration Examples
# ============================================================================

async def main() -> None:
    """Launch with declarative config."""

    # Example 1: Default pose tracking (right wrist)
    config = LaunchConfig(detector_type="pose")

    # Example 2: Track different body part
    # config = LaunchConfig(
    #     detector_type="pose",
    #     pose_detector=PoseDetectorParams(
    #         target_keypoint=CocoKeypoint.NOSE,  # Track nose instead
    #         detection_threshold=0.4,
    #         keypoint_threshold=0.4,
    #     ),
    #     motor=MotorNodeParams(gain=0.08, deadzone=20),
    # )

    # Example 3: Track left hand
    # config = LaunchConfig(
    #     detector_type="pose",
    #     pose_detector=PoseDetectorParams(
    #         target_keypoint=CocoKeypoint.LEFT_WRIST,
    #     ),
    # )

    # Example 4: Brightness detector (original behavior)
    # config = LaunchConfig(
    #     detector_type="brightness",
    #     brightness_detector=BrightnessDetectorParams(blur_size=21, threshold=150),
    #     motor=MotorNodeParams(gain=0.08, deadzone=20),
    # )

    launcher = Launcher(config)
    await launcher.run()


if __name__ == "__main__":
    asyncio.run(main())