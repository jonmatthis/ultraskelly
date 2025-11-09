import asyncio
import logging

from pydantic import BaseModel, Field, SkipValidation, ConfigDict

logger = logging.getLogger(__name__)
from ultraskelly.core.bot.base_abcs import DetectorType, Node, NodeParams
from ultraskelly.core.bot.motor.head_node import HeadNode, HeadNodeParams
from ultraskelly.core.bot.sensory.bright_point_detection_node import (
    BrightnessDetectorNode,
    BrightnessDetectorParams,
)
from ultraskelly.core.bot.sensory.camera_node import VisionNode, VisionNodeParams
from ultraskelly.core.bot.sensory.pose_detection_node import (
    PoseDetectorNode,
    PoseDetectorParams,
)
from ultraskelly.ui.cv2_ui import UINode, UINodeParams
from ultraskelly.core.pubsub.pubsub_manager import (
    PubSubTopicManager,
    get_or_create_pipeline_pubsub_manager,
)


class LaunchConfig(NodeParams):
    """Declarative launch configuration."""

    detector_type: DetectorType = Field(
        default=DetectorType.POSE, description="Which detector to use"
    )
    vision: VisionNodeParams = Field(default_factory=VisionNodeParams)
    brightness_detector: BrightnessDetectorParams = Field(
        default_factory=BrightnessDetectorParams
    )
    pose_detector: PoseDetectorParams = Field(default_factory=PoseDetectorParams)
    head: HeadNodeParams = Field(default_factory=HeadNodeParams)
    ui: UINodeParams = Field(default_factory=UINodeParams)


class BotLauncher(BaseModel):
    """Main launcher for the bot system."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: LaunchConfig
    pubsub: SkipValidation[PubSubTopicManager]
    nodes: list[Node] = Field(default_factory=list)
    tasks: list[asyncio.Task] = Field(default_factory=list, exclude=True)

    @classmethod
    def from_config(cls, config: LaunchConfig) -> "BotLauncher":
        """Instantiate nodes based on config."""
        logger.info("Creating nodes from launch config...")

        launcher = cls(config=config, pubsub=get_or_create_pipeline_pubsub_manager())

        # Create motor and UI nodes
        launcher.nodes.append(HeadNode.create(pubsub=launcher.pubsub, params=config.head))
        launcher.nodes.append(UINode.create(pubsub=launcher.pubsub, params=config.ui))

        # Create detector based on type
        if config.detector_type == DetectorType.BRIGHTNESS:
            launcher.nodes.append(
                VisionNode.create(pubsub=launcher.pubsub, params=config.vision)
            )
            launcher.nodes.append(
                BrightnessDetectorNode.create(
                    pubsub=launcher.pubsub, params=config.brightness_detector
                )
            )
        elif config.detector_type == DetectorType.POSE:
            launcher.nodes.append(
                PoseDetectorNode.create(pubsub=launcher.pubsub, params=config.pose_detector)
            )
        else:
            raise ValueError(f"Unknown detector type: {config.detector_type}")

        logger.info(f"Created {len(launcher.nodes)} nodes")
        return launcher

    async def run(self) -> None:
        """Launch all nodes."""
        logger.info("=" * 60)
        logger.info("LAUNCHING TARGET TRACKER WITH SOPHISTICATED PUBSUB")
        logger.info("=" * 60)

        try:
            # Start all nodes as async tasks
            for node in self.nodes:
                task = asyncio.create_task(node.run())
                self.tasks.append(task)

            # Wait for all tasks
            await asyncio.gather(*self.tasks)

        except KeyboardInterrupt:
            logger.info("\nStopping...")
        finally:
            # Stop all nodes
            for node in self.nodes:
                node.stop()

            # Cancel all tasks
            for task in self.tasks:
                task.cancel()

            # Wait for tasks to finish cancelling
            await asyncio.gather(*self.tasks, return_exceptions=True)

            # Clean up pubsub
            self.pubsub.close()
            logger.info("All nodes stopped and pubsub closed")