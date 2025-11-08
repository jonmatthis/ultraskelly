import asyncio
import time

import numpy as np
from pydantic import BaseModel, Field

from ultraskelly.core.bot.__main__bot import logger
from ultraskelly.core.bot.sensory.camera_node import FrameMessage
from ultraskelly.core.bot.motor.head_node import TargetLocationMessage, ServoStateMessage
from ultraskelly.core.bot.sensory.pose_detection_node import PoseDataMessage, SKELETON_CONNECTIONS
from ultraskelly.core.bot.pubsub import PubSub


class UINodeParams(BaseModel):
    """Parameters for UINode."""
    deadzone: int = Field(default=30, ge=0)
    window_name: str = Field(default="Async Target Tracker")


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
