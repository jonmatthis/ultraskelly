import logging
import time

import numpy as np


from pydantic import Field, SkipValidation

from ultraskelly import FAIL_ON_IMPORTS
from ultraskelly.core.bot.base_abcs import Node, NodeParams
from ultraskelly.core.pubsub.bot_topics import (
    ServoStateMessage,
    ServoStateTopic,
    TargetLocationMessage,
    TargetLocationTopic,
)
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager

logger = logging.getLogger(__name__)
try:
    from adafruit_servokit import ServoKit
except ImportError:
    if FAIL_ON_IMPORTS:
        raise
    ServoKit = None  # type: ignore
    logger.warning("Could not import adafruit_servokit. Servo control will not work.")

class MotorNodeParams(NodeParams):
    """Parameters for MotorNode."""

    pan_channel: int = Field(default=11, ge=0, le=15)
    tilt_channel: int = Field(default=3, ge=0, le=15)
    roll_channel: int = Field(default=7, ge=0, le=15)
    target_x: int = Field(default=320, ge=0)
    target_y: int = Field(default=240, ge=0)
    offset_x_ratio: float = Field(default=-0.2, ge=-1.0, le=1.0)
    offset_y_ratio: float = Field(default=0.2, ge=-1.0, le=1.0)
    gain: float = Field(default=0.05, gt=0.0, le=1.0)
    roll_gain: float = Field(default=0.3, gt=0.0, le=1.0)
    roll_smoothing: float = Field(default=0.7, ge=0.0, le=1.0)
    deadzone: int = Field(default=30, ge=0)
    roll_deadzone: float = Field(default=5.0, ge=0.0)

    @property
    def x_offset(self) -> int:
        return int(self.offset_x_ratio * self.target_x * 2)

    @property
    def y_offset(self) -> int:
        return int(self.offset_y_ratio * self.target_y * 2)


class MotorNode(Node):
    """Generic servo controller - tracks whatever target is published."""

    params: MotorNodeParams
    kit: SkipValidation[ServoKit] = Field(default=None, exclude=True)
    pan_angle: float = Field(default=90.0)
    tilt_angle: float = Field(default=90.0)
    roll_angle: float = Field(default=90.0)
    smoothed_target_roll: float | None = Field(default=None)
    target_subscription: SkipValidation[object] = Field(default=None, exclude=True)

    @classmethod
    def create(cls, *, pubsub: PubSubTopicManager, params: MotorNodeParams) -> "MotorNode":
        """Factory method to create and initialize MotorNode."""
        node = cls(pubsub=pubsub, params=params)

        # Initialize servos
        node.kit = ServoKit(channels=16)
        node.kit.servo[params.pan_channel].angle = node.pan_angle
        node.kit.servo[params.tilt_channel].angle = node.tilt_angle
        node.kit.servo[params.roll_channel].angle = node.roll_angle

        # Get subscription
        node.target_subscription = pubsub.get_subscription(TargetLocationTopic)

        return node

    def _update_servos(self, *, msg: TargetLocationMessage) -> ServoStateMessage:
        """Process target message and update servos."""
        if msg.x is None or msg.y is None:
            return ServoStateMessage(
                pan_angle=self.pan_angle,
                tilt_angle=self.tilt_angle,
                roll_angle=self.roll_angle,
                is_locked_x=False,
                is_locked_y=False,
                is_locked_roll=False,
            )

        # Calculate errors
        error_x = msg.x - self.params.target_x - self.params.x_offset
        error_y = msg.y - self.params.target_y - self.params.y_offset

        # Check lock status
        is_locked_x = abs(error_x) <= self.params.deadzone
        is_locked_y = abs(error_y) <= self.params.deadzone

        # Update pan/tilt servos
        if not is_locked_x:
            self.pan_angle += error_x * self.params.gain
            self.pan_angle = float(np.clip(self.pan_angle, 0.0, 180.0))
            self.kit.servo[self.params.pan_channel].angle = self.pan_angle

        if not is_locked_y:
            self.tilt_angle += error_y * self.params.gain
            self.tilt_angle = float(np.clip(self.tilt_angle, 0.0, 180.0))
            self.kit.servo[self.params.tilt_channel].angle = self.tilt_angle

        # Handle roll angle
        is_locked_roll = False
        if msg.angle is not None:
            target_roll = 90.0 + msg.angle

            if self.smoothed_target_roll is None:
                self.smoothed_target_roll = target_roll
            else:
                jump = abs(target_roll - self.smoothed_target_roll)
                if jump > 90.0:
                    self.smoothed_target_roll = target_roll
                else:
                    self.smoothed_target_roll = (
                        self.params.roll_smoothing * self.smoothed_target_roll
                        + (1.0 - self.params.roll_smoothing) * target_roll
                    )

            roll_error = self.smoothed_target_roll - self.roll_angle
            is_locked_roll = abs(roll_error) <= self.params.roll_deadzone

            if not is_locked_roll:
                self.roll_angle += roll_error * self.params.roll_gain
                self.roll_angle = float(np.clip(self.roll_angle, 0.0, 180.0))
                self.kit.servo[self.params.roll_channel].angle = self.roll_angle

        return ServoStateMessage(
            pan_angle=self.pan_angle,
            tilt_angle=self.tilt_angle,
            roll_angle=self.roll_angle,
            is_locked_x=is_locked_x,
            is_locked_y=is_locked_y,
            is_locked_roll=is_locked_roll,
        )

    def run(self) -> None:
        """Main motor control loop."""
        logger.info(f"Starting MotorNode [gain={self.params.gain}]")

        try:
            while not self.stop_event.is_set():
                try:
                    # Get message from subscription queue
                    if not self.target_subscription.empty():
                        msg = self.target_subscription.get(timeout=0.1)
                        state = self._update_servos(msg=msg)
                        self.pubsub.publish(ServoStateTopic, state)
                except Exception:
                    continue
        finally:
            # Center servos
            self.kit.servo[self.params.pan_channel].angle = 90.0
            self.kit.servo[self.params.tilt_channel].angle = 90.0
            self.kit.servo[self.params.roll_channel].angle = 90.0
            time.sleep(0.5)
            self.kit.servo[self.params.pan_channel].angle = None
            self.kit.servo[self.params.tilt_channel].angle = None
            self.kit.servo[self.params.roll_channel].angle = None
            logger.info("MotorNode stopped")