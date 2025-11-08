"""
Async Target Tracker with Pose Estimation Integration

ROS2-inspired architecture: independent nodes + parameter system + launch config.
Now includes IMX500-based human pose tracking.
"""
import asyncio
import logging

import cv2
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

from ultraskelly.core.bot.bot_launcher import LaunchConfig, Launcher
from ultraskelly.core.bot.motor.head_node import MotorNodeParams
from ultraskelly.core.bot.sensory.pose_detection_node import PoseDetectorParams, CocoKeypoint

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def main() -> None:
    """Launch with declarative config."""

    # Example 1: Default pose tracking (right wrist)

    config = LaunchConfig(
        detector_type="pose",
        pose_detector=PoseDetectorParams(
            target_keypoint=CocoKeypoint.NOSE,  # Track nose instead
            detection_threshold=0.4,
            keypoint_threshold=0.4,
        ),
        motor=MotorNodeParams(gain=0.08, deadzone=40),
    )

    # Example : Track left hand
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
