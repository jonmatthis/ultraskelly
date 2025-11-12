import asyncio
from collections import deque

import numpy as np
from adafruit_motor.motor import DCMotor
from adafruit_motorkit import MotorKit
from pydantic import Field, SkipValidation

from ultraskelly.core.bot.base_abcs import NodeParams, Node
from ultraskelly.core.pubsub.bot_topics import ServoStateTopic, ServoStateMessage
from ultraskelly.core.pubsub.pubsub_abcs import TopicSubscriptionQueue
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager

import  logging
logger = logging.getLogger(__name__)

class WaistNodeParams(NodeParams):
    waist_motor_channel: int = Field(default=1, ge=1, le=4)
    waist_deadzone: float = Field(default=10.0, ge=0.0, le=90.0)


class WaistNode(Node):
    """Generic servo controller - tracks whatever target is published."""

    params: WaistNodeParams
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    waist_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    waist_motor_throttle: float = Field(default=0, ge=-1.0, le=1.0)
    head_servo_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)

    @classmethod
    def create(cls, *, pubsub: PubSubTopicManager, params: WaistNodeParams) -> "WaistNode":
        """Factory method to create and initialize WaistNode."""

        motor_kit = MotorKit(address=0x60, pwm_frequency=1600)
        waist_motor: DCMotor|None = None
        match params.waist_motor_channel:
            case 1:
                waist_motor = motor_kit.motor1
            case 2:
                waist_motor = motor_kit.motor2
            case 3:
                waist_motor = motor_kit.motor3
            case 4:
                waist_motor = motor_kit.motor4
            case _:
                raise ValueError(f"Invalid waist motor channel: {params.waist_motor_channel}")

        if waist_motor is None:
            raise ValueError(f"Could not initialize waist motor on channel {params.waist_motor_channel}")
        # Get subscription
        head_servo_subscription = pubsub.get_subscription(ServoStateTopic)
        return cls(pubsub=pubsub,
                   params=params,
                   motor_kit=motor_kit,
                   waist_motor=waist_motor,
                   head_servo_subscription=head_servo_subscription)


    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(f"Starting WaistNode [deadzone={self.params.waist_deadzone}]")
        recent_head_offsets = deque(maxlen=100)
        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.01)
                try:
                    # Get message from subscription queue with timeout
                    if not self.head_servo_subscription.empty():
                        msg: ServoStateMessage =  await self.head_servo_subscription.get()
                        head_pan_off_center = msg.pan_angle - 90.0
                        offset_changing = np.mean(np.diff(recent_head_offsets))> self.params.waist_deadzone if len(recent_head_offsets) > 1 else True
                        if abs(head_pan_off_center) > self.params.waist_deadzone and offset_changing:
                            if head_pan_off_center > 0:
                                # Target is to the right, turn waist to the right
                                self.waist_motor.throttle = 1.0
                            else:
                                # Target is to the left, turn waist to the left
                                self.waist_motor.throttle = -1.0
                        else:
                            self.waist_motor.throttle = 0.0
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue
        finally:
            # Center servos
            self.waist_motor.throttle = 0.0
            await asyncio.sleep(0.5)
            logger.info('WaistNode stopped.')