"""
Target Tracker with Pose Estimation Integration

ROS2-inspired architecture: independent nodes + parameter system + launch config.
Now includes IMX500-based human pose tracking.
"""
import logging
logger = logging.getLogger(__name__)
from ultraskelly.core.bot.base_abcs import DetectorType
from ultraskelly.bot_launcher import LaunchConfig, BotLauncher
from ultraskelly.core.bot.motor.head_node import MotorNodeParams
from ultraskelly.core.bot.sensory.pose_detection_node import PoseDetectorParams, CocoKeypoint

logger = logging.getLogger(__name__)


def main() -> None:
    """Launch with declarative config."""

    # Example 1: Default pose tracking (nose)
    config = LaunchConfig(
        detector_type=DetectorType.POSE,
        pose_detector=PoseDetectorParams(
            target_keypoint=CocoKeypoint.NOSE,
            detection_threshold=0.4,
            keypoint_threshold=0.4,
        ),
        motor=MotorNodeParams(gain=0.08, deadzone=40),
    )

    # Example : Track left wrist
    # config = LaunchConfig(
    #     detector_type=DetectorType.POSE,
    #     pose_detector=PoseDetectorParams(
    #         target_keypoint=CocoKeypoint.LEFT_WRIST,
    #     ),
    # )

    # Example : Brightness detector
    # config = LaunchConfig(
    #     detector_type=DetectorType.BRIGHTNESS,
    #     brightness_detector=BrightnessDetectorParams(blur_size=21, threshold=150),
    #     motor=MotorNodeParams(gain=0.08, deadzone=20),
    # )

    launcher = BotLauncher.from_config(config)
    launcher.run()


if __name__ == "__main__":
    main()