import asyncio

from adafruit_motor.motor import DCMotor
from adafruit_motorkit import MotorKit
from pydantic import Field, SkipValidation

from ultraskelly.core.bot.base_abcs import NodeParams, Node
from ultraskelly.core.pubsub.bot_topics import ServoStateTopic, ServoStateMessage
from ultraskelly.core.pubsub.pubsub_abcs import TopicSubscriptionQueue
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager

import  logging
logger = logging.getLogger(__name__)

class MouthNodeParams(NodeParams):
    mouth_motor_channel: int = Field(default=2, ge=1, le=4)
    head_servo_state_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)


class MouthNode(Node):
    """Generic servo controller - tracks whatever target is published."""

    params: MouthNodeParams
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    mouth_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    mouth_motor_throttle: float = Field(default=0, ge=-1.0, le=1.0)
    head_servo_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)

    @classmethod
    def create(cls, *, pubsub: PubSubTopicManager, params: MouthNodeParams) -> "MouthNode":
        """Factory method to create and initialize MouthNode."""

        motor_kit = MotorKit(address=0x60, pwm_frequency=1600)
        mouth_motor: DCMotor|None = None
        match params.mouth_motor_channel:
            case 1:
                mouth_motor = motor_kit.motor1
            case 2:
                mouth_motor = motor_kit.motor2
            case 3:
                mouth_motor = motor_kit.motor3
            case 4:
                mouth_motor = motor_kit.motor4
            case _:
                raise ValueError(f"Invalid mouth motor channel: {params.mouth_motor_channel}")

        if mouth_motor is None:
            raise ValueError(f"Could not initialize mouth motor on channel {params.mouth_motor_channel}")
        # Get subscription
        head_servo_subscription = pubsub.get_subscription(ServoStateTopic)
        return cls(pubsub=pubsub,
                   params=params,
                   motor_kit=motor_kit,
                   mouth_motor=mouth_motor,
                   head_servo_subscription=head_servo_subscription)


    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(f"Starting HeadNode [gain={self.params.gain}]")

        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.01)
                try:
                    # Get message from subscription queue with timeout
                    if not self.head_servo_subscription.empty():
                        msg: ServoStateMessage =  await self.head_servo_subscription.get()
                        if msg.is_locked_x or msg.is_locked_y:
                            self.mouth_motor.throttle = 1.0
                        else:
                            self.mouth_motor.throttle = 0.0
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue
        finally:
            # Center servos
            self.mouth_motor.throttle = 0.0
            await asyncio.sleep(0.5)
            logger.info('MouthNode stopped.')