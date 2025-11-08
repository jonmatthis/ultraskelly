import asyncio
import time
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from ultraskelly.core.bot.__main__bot import logger
from ultraskelly.core.bot.pubsub import PubSub


class VisionNodeParams(BaseModel):
    """Parameters for VisionNode."""
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)


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


@dataclass(frozen=True)
class FrameMessage:
    """Raw frame from camera."""
    frame: np.ndarray
    timestamp: float
