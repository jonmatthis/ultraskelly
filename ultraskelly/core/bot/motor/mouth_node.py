import asyncio
import time
import logging
from enum import Enum

from adafruit_motor.motor import DCMotor
from adafruit_motorkit import MotorKit
from pydantic import Field, SkipValidation

from ultraskelly.core.bot.base_abcs import NodeParams, Node
from ultraskelly.core.pubsub.bot_topics import ServoStateTopic, ServoStateMessage
from ultraskelly.core.pubsub.pubsub_abcs import TopicSubscriptionQueue
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager

logger = logging.getLogger(__name__)


class MouthState(Enum):
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"


class MouthNodeParams(NodeParams):
    mouth_motor_channel: int = Field(default=2, ge=1, le=4)
    open_duration: float = Field(default=2.0, description="Time to keep mouth open before closing")
    close_ramp_duration: float = Field(default=1.5, description="Time to ramp down throttle when closing")


class MouthNode(Node):
    """Mouth controller that slowly closes after opening."""

    params: MouthNodeParams
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    mouth_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    head_servo_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)

    # State tracking
    mouth_state: MouthState = Field(default=MouthState.CLOSED)
    state_start_time: float = Field(default=0.0)
    current_throttle: float = Field(default=0.0)

    @classmethod
    def create(cls, *, pubsub: PubSubTopicManager, params: MouthNodeParams) -> "MouthNode":
        """Factory method to create and initialize MouthNode."""
        motor_kit = MotorKit(address=0x60, pwm_frequency=1600)

        # Select motor channel
        mouth_motor: DCMotor | None = None
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

        logger.info("Testing mouth motor...")
        mouth_motor.throttle = 1.0
        time.sleep(0.5)
        mouth_motor.throttle = -1.0
        time.sleep(0.5)
        mouth_motor.throttle = 0.0

        head_servo_subscription = pubsub.get_subscription(ServoStateTopic)

        return cls(
            pubsub=pubsub,
            params=params,
            motor_kit=motor_kit,
            mouth_motor=mouth_motor,
            head_servo_subscription=head_servo_subscription
        )

    def _update_mouth_state(self, *, is_locked: bool) -> None:
        """Update mouth state machine based on head lock status."""
        current_time = time.time()

        # State machine logic
        if self.mouth_state == MouthState.CLOSED:
            if is_locked:
                self.mouth_state = MouthState.OPENING
                self.state_start_time = current_time
                self.current_throttle = -1.0

        elif self.mouth_state == MouthState.OPENING:
            if not is_locked:
                self.mouth_state = MouthState.CLOSING
                self.state_start_time = current_time
            elif current_time - self.state_start_time > 0.2:  # Quick open
                self.mouth_state = MouthState.OPEN
                self.state_start_time = current_time

        elif self.mouth_state == MouthState.OPEN:
            if not is_locked:
                self.mouth_state = MouthState.CLOSING
                self.state_start_time = current_time
            elif current_time - self.state_start_time > self.params.open_duration:
                self.mouth_state = MouthState.CLOSING
                self.state_start_time = current_time

        elif self.mouth_state == MouthState.CLOSING:
            elapsed = current_time - self.state_start_time
            if elapsed >= self.params.close_ramp_duration:
                self.mouth_state = MouthState.CLOSED
                self.current_throttle = 0.0
            else:
                # Ramp down throttle
                progress = elapsed / self.params.close_ramp_duration
                self.current_throttle = -1.0 * (1.0 - progress)

            # If head locks again while closing, reopen
            if is_locked and self.current_throttle > -0.5:
                self.mouth_state = MouthState.OPENING
                self.state_start_time = current_time

    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(f"Starting MouthNode [open_duration={self.params.open_duration}s]")

        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.01)

                # Process head servo messages
                is_locked = False
                while not self.head_servo_subscription.empty():
                    msg: ServoStateMessage = await self.head_servo_subscription.get()
                    is_locked = msg.is_locked_x and msg.is_locked_y

                # Update state machine
                self._update_mouth_state(is_locked=is_locked)

                # Apply throttle based on state
                if self.mouth_state == MouthState.OPENING or self.mouth_state == MouthState.OPEN:
                    self.current_throttle = -1.0
                elif self.mouth_state == MouthState.CLOSED:
                    self.current_throttle = 0.0
                # CLOSING state updates throttle in _update_mouth_state

                self.mouth_motor.throttle = self.current_throttle

        finally:
            self.mouth_motor.throttle = 0.0
            await asyncio.sleep(0.5)
            logger.info('MouthNode stopped.')