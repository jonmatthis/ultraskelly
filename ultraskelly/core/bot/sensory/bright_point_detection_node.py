import time

import numpy as np
from pydantic import BaseModel, Field

from ultraskelly.core.bot.__main__bot import logger
from ultraskelly.core.bot.sensory.camera_node import FrameMessage
from ultraskelly.core.bot.motor.head_node import TargetLocationMessage
from ultraskelly.core.bot.pubsub import PubSub


class BrightnessDetectorParams(BaseModel):
    """Parameters for BrightnessDetectorNode."""
    blur_size: int = Field(default=15, ge=1, description="Must be odd")
    threshold: int = Field(default=100, ge=0, le=255)


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
