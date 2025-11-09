from adafruit_motor.motor import DCMotor
from adafruit_motorkit import MotorKit

from ultraskelly.core.bot.base_abcs import NodeParams, Node
from pydantic import Field, SkipValidation

from ultraskelly.core.pubsub.bot_topics import ServoStateTopic
from ultraskelly.core.pubsub.pubsub_abcs import TopicSubscriptionQueue
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager


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
        mouth_motor = motor_kit.motor

        # Get subscription
        head_servo_subscription = pubsub.get_subscription(ServoStateTopic)
        return cls(pubsub=pubsub,
                   params=params)
