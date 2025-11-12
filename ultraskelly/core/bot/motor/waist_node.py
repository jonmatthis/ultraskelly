import asyncio
import logging
from collections import deque

import numpy as np
from adafruit_motor.motor import DCMotor
from adafruit_motorkit import MotorKit
from pydantic import Field, SkipValidation

from ultraskelly.core.bot.base_abcs import NodeParams, Node
from ultraskelly.core.pubsub.bot_topics import ServoStateTopic, ServoStateMessage
from ultraskelly.core.pubsub.pubsub_abcs import TopicSubscriptionQueue
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager

logger = logging.getLogger(__name__)


class WaistNodeParams(NodeParams):
    waist_motor_channel: int = Field(default=1, ge=1, le=4)

    # Hysteresis thresholds - turn on at larger angle, turn off at smaller
    activation_threshold: float = Field(default=15.0, ge=0.0, le=90.0,
                                        description="Pan angle offset to start waist movement")
    deactivation_threshold: float = Field(default=8.0, ge=0.0, le=90.0,
                                          description="Pan angle offset to stop waist movement")

    # Control parameters
    proportional_gain: float = Field(default=0.015, gt=0.0, le=0.1,
                                     description="P-gain for waist control")
    max_throttle: float = Field(default=0.6, gt=0.0, le=1.0,
                                description="Maximum throttle to prevent wild movements")

    # Smoothing parameters
    angle_smoothing_window: int = Field(default=10, ge=1, le=50,
                                        description="Window size for angle smoothing")
    throttle_ramp_rate: float = Field(default=0.05, gt=0.0, le=0.5,
                                      description="Max throttle change per update")


class WaistNode(Node):
    """Waist controller that nulls head pan rotation with damping and hysteresis."""

    params: WaistNodeParams
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    waist_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    head_servo_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)

    # State tracking
    is_active: bool = Field(default=False, description="Whether waist control is active")
    current_throttle: float = Field(default=0.0)
    target_throttle: float = Field(default=0.0)
    pan_angle_buffer: deque[float] = Field(default_factory=lambda: deque(maxlen=10))
    smoothed_pan_offset: float = Field(default=0.0)

    @classmethod
    def create(cls, *, pubsub: PubSubTopicManager, params: WaistNodeParams) -> "WaistNode":
        """Factory method to create and initialize WaistNode."""
        motor_kit = MotorKit(address=0x60, pwm_frequency=1600)

        # Select motor channel
        waist_motor: DCMotor | None = None
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

        head_servo_subscription = pubsub.get_subscription(ServoStateTopic)

        # Create instance with proper buffer size
        instance = cls(
            pubsub=pubsub,
            params=params,
            motor_kit=motor_kit,
            waist_motor=waist_motor,
            head_servo_subscription=head_servo_subscription
        )
        instance.pan_angle_buffer = deque(maxlen=params.angle_smoothing_window)

        return instance

    def _calculate_control(self, *, pan_angle: float) -> float:
        """Calculate waist control with hysteresis and damping."""
        # Calculate offset from center (90 degrees)
        pan_offset = pan_angle - 90.0

        # Add to smoothing buffer
        self.pan_angle_buffer.append(pan_offset)

        # Calculate smoothed offset
        if len(self.pan_angle_buffer) > 0:
            self.smoothed_pan_offset = float(np.mean(self.pan_angle_buffer))
        else:
            self.smoothed_pan_offset = pan_offset

        abs_offset = abs(self.smoothed_pan_offset)

        # Hysteresis logic
        if not self.is_active:
            # Activate if offset exceeds activation threshold
            if abs_offset > self.params.activation_threshold:
                self.is_active = True
                logger.debug(f"Waist control activated at offset: {self.smoothed_pan_offset:.1f}°")
        else:
            # Deactivate if offset falls below deactivation threshold
            if abs_offset < self.params.deactivation_threshold:
                self.is_active = False
                logger.debug(f"Waist control deactivated at offset: {self.smoothed_pan_offset:.1f}°")

        # Calculate target throttle
        if self.is_active:
            # Proportional control with saturation
            raw_throttle = -self.smoothed_pan_offset * self.params.proportional_gain
            self.target_throttle = float(np.clip(raw_throttle, -self.params.max_throttle, self.params.max_throttle))
        else:
            self.target_throttle = 0.0

        # Apply throttle ramping for smooth acceleration/deceleration
        throttle_diff = self.target_throttle - self.current_throttle
        max_change = self.params.throttle_ramp_rate

        if abs(throttle_diff) > max_change:
            change = max_change if throttle_diff > 0 else -max_change
            self.current_throttle += change
        else:
            self.current_throttle = self.target_throttle

        return self.current_throttle

    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(
            f"Starting WaistNode [activation={self.params.activation_threshold}°, "
            f"deactivation={self.params.deactivation_threshold}°, "
            f"gain={self.params.proportional_gain}, "
            f"max_throttle={self.params.max_throttle}]"
        )

        try:
            latest_pan_angle: float | None = None

            while not self.stop_event.is_set():
                await asyncio.sleep(0.01)  # 100Hz update rate

                # Get latest head servo state
                while not self.head_servo_subscription.empty():
                    msg: ServoStateMessage = await self.head_servo_subscription.get()
                    latest_pan_angle = msg.pan_angle

                # Calculate and apply control
                if latest_pan_angle is not None:
                    throttle = self._calculate_control(pan_angle=latest_pan_angle)
                    self.waist_motor.throttle = throttle

                    # Log state changes for debugging
                    if abs(throttle) > 0.01 and abs(self.current_throttle - throttle) > 0.01:
                        logger.trace(
                            f"Waist: offset={self.smoothed_pan_offset:.1f}°, "
                            f"throttle={throttle:.2f}, active={self.is_active}"
                        )

        finally:
            # Ensure motor stops
            self.waist_motor.throttle = 0.0
            await asyncio.sleep(0.5)
            logger.info('WaistNode stopped.')