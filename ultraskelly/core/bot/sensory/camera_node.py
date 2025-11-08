import logging
import time

from pydantic import Field, SkipValidation

from ultraskelly import FAIL_ON_IMPORTS

logger = logging.getLogger(__name__)

from ultraskelly.core.bot.base_abcs import  Node, NodeParams
from ultraskelly.core.pubsub.bot_topics import FrameMessage
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager

try:
    from picamera2 import Picamera2
except ImportError:
    if FAIL_ON_IMPORTS:
        raise
    logger.error("Picamera2 library not found - vision node will not work.")
    Picamera2 = None  # type: ignore

class VisionNodeParams(NodeParams):
    """Parameters for VisionNode."""

    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)




class VisionNode(Node):
    """Captures frames and publishes them."""

    params: VisionNodeParams
    picam2: SkipValidation[Picamera2] = Field(default=None, exclude=True)

    @classmethod
    def create(cls, *, pubsub: PubSubTopicManager, params: VisionNodeParams) -> "VisionNode":
        """Factory method to create and initialize VisionNode."""
        node = cls(pubsub=pubsub, params=params)

        # Initialize camera
        node.picam2 = Picamera2()
        config = node.picam2.create_preview_configuration(
            main={"size": (params.width, params.height), "format": "RGB888"}
        )
        node.picam2.configure(config)

        return node

    def run(self) -> None:
        """Main vision node loop."""
        logger.info(f"Starting VisionNode [{self.params.width}x{self.params.height}]")
        self.picam2.start()
        time.sleep(1)

        try:
            while not self.stop_event.is_set():
                frame = self.picam2.capture_array()
                self.pubsub.frame.publish(FrameMessage(frame=frame, timestamp=time.time()))
                time.sleep(0.001)
        finally:
            self.picam2.stop()
            logger.info("VisionNode stopped")