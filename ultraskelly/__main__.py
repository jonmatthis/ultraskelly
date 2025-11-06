"""
Async Target Tracker with Parameters and Declarative Launch System

ROS2-inspired architecture: independent nodes + parameter system + launch config.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from adafruit_servokit import ServoKit
from picamera2 import Picamera2
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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
# Vision Node
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
        # Also need to account for which axis is major
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
                    # -90° rotation -> 0° servo, 0° rotation -> 90° servo, +90° rotation -> 180° servo
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

        self.latest_target: TargetLocationMessage | None = None
        self.latest_servo_state: ServoStateMessage | None = None

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def _draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking visualization on frame."""
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

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
                end_y = int(y - line_length * np.cos(angle_rad))  # Negative because y-axis points down

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
                f"Target Rot: {self.latest_target.angle:.1f}°",
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

                cv2.imshow(self.params.window_name, vis_frame)

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

    detector_type: Literal["brightness"] = Field(
        default="brightness",
        description="Which detector to use"
    )

    vision: VisionNodeParams = Field(default_factory=VisionNodeParams)
    brightness_detector: BrightnessDetectorParams = Field(default_factory=BrightnessDetectorParams)
    motor: MotorNodeParams = Field(default_factory=MotorNodeParams)
    ui: UINodeParams = Field(default_factory=UINodeParams)


class Launcher:
    """Launches nodes based on declarative config."""

    def __init__(self, config: LaunchConfig) -> None:
        self.config = config
        self.pubsub = PubSub()
        self.nodes: list[VisionNode | BrightnessDetectorNode | MotorNode | UINode] = []

    def _create_nodes(self) -> None:
        """Instantiate nodes based on config."""
        logger.info("Creating nodes from launch config...")

        # Always create vision, motor, and UI
        self.nodes.append(VisionNode(self.pubsub, self.config.vision))
        self.nodes.append(MotorNode(self.pubsub, self.config.motor))
        self.nodes.append(UINode(self.pubsub, self.config.ui))

        # Create detector based on type
        if self.config.detector_type == "brightness":
            self.nodes.append(
                BrightnessDetectorNode(self.pubsub, self.config.brightness_detector)
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

    # Example 1: Default config
    config = LaunchConfig()

    # Example 2: Custom parameters
    # config = LaunchConfig(
    #     detector_type="brightness",
    #     vision=VisionNodeParams(width=640, height=480),
    #     brightness_detector=BrightnessDetectorParams(blur_size=21, threshold=150),
    #     motor=MotorNodeParams(gain=0.08, deadzone=20, roll_gain=0.5, roll_deadzone=10.0),
    #     ui=UINodeParams(window_name="My Custom Tracker")
    # )

    # Example 3: Load from file
    # import json
    # with open("launch_config.json") as f:
    #     config = LaunchConfig(**json.load(f))

    launcher = Launcher(config)
    await launcher.run()


if __name__ == "__main__":
    asyncio.run(main())