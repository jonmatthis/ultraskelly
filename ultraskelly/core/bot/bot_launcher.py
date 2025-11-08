import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from ultraskelly.core.bot.__main__bot import logger
from ultraskelly.core.bot.sensory.bright_point_detection_node import BrightnessDetectorParams, BrightnessDetectorNode
from ultraskelly.core.bot.motor.head_node import MotorNodeParams, MotorNode
from ultraskelly.core.bot.sensory.camera_node import VisionNodeParams, VisionNode
from ultraskelly.core.bot.sensory.pose_detection_node import PoseDetectorParams, PoseDetectorNode
from ultraskelly.core.bot.pubsub import PubSub
from ultraskelly.core.bot.ui import UINodeParams, UINode


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
