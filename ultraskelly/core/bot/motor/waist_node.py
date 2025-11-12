import asyncio
import logging
import time
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

    # Motor control parameters
    min_throttle: float = Field(default=0.7, ge=0.5, le=1.0,
                                description="Minimum throttle to overcome static friction")
    max_throttle: float = Field(default=1.0, ge=0.7, le=1.0,
                                description="Maximum throttle limit")
    proportional_gain: float = Field(default=0.015, gt=0.0, le=0.1,
                                     description="P-gain for waist control")

    # Safety parameters
    max_active_duration: float = Field(default=3.0, gt=0.0,
                                       description="Maximum time to keep motor active before timeout")
    cooldown_duration: float = Field(default=2.0, gt=0.0,
                                     description="Cooldown period after timeout")

    # Smoothing parameters
    angle_smoothing_window: int = Field(default=10, ge=1, le=50,
                                        description="Window size for angle smoothing")


class WaistNode(Node):
    """Waist controller with minimum throttle threshold and timeout protection."""

    params: WaistNodeParams
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    waist_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    head_servo_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)

    # State tracking
    is_active: bool = Field(default=False, description="Whether waist control is active")
    activation_start_time: float | None = Field(default=None)
    in_cooldown: bool = Field(default=False)
    cooldown_start_time: float | None = Field(default=None)

    # Control values
    current_throttle: float = Field(default=0.0)
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

    def _map_throttle(self, *, control_value: float) -> float:
        """Map control value to motor throttle with minimum threshold.

        Maps control values to either 0.0 or [min_throttle, max_throttle] range.
        """
        if abs(control_value) < 0.001:  # Essentially zero
            return 0.0

        # Map to minimum throttle + proportional range
        throttle_range = self.params.max_throttle - self.params.min_throttle
        sign = 1.0 if control_value > 0 else -1.0

        # Scale control value (0 to 1) to throttle range
        scaled_throttle = self.params.min_throttle + abs(control_value) * throttle_range

        # Apply sign and clamp
        final_throttle = sign * scaled_throttle
        return float(np.clip(final_throttle, -self.params.max_throttle, self.params.max_throttle))

    def _check_timeout(self) -> bool:
        """Check if motor has been active too long and needs timeout."""
        if not self.is_active or self.activation_start_time is None:
            return False

        current_time = time.time()
        active_duration = current_time - self.activation_start_time

        if active_duration > self.params.max_active_duration:
            logger.warning(f"Waist motor timeout after {active_duration:.1f}s")
            return True

        return False

    def _check_cooldown(self) -> bool:
        """Check if still in cooldown period."""
        if not self.in_cooldown or self.cooldown_start_time is None:
            return False

        current_time = time.time()
        cooldown_elapsed = current_time - self.cooldown_start_time

        if cooldown_elapsed > self.params.cooldown_duration:
            self.in_cooldown = False
            self.cooldown_start_time = None
            logger.info("Waist motor cooldown complete")
            return False

        return True

    def _calculate_control(self, *, pan_angle: float) -> float:
        """Calculate waist control with hysteresis, minimum throttle, and timeout."""
        # Check cooldown first
        if self._check_cooldown():
            return 0.0

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
                self.activation_start_time = time.time()
                logger.debug(f"Waist control activated at offset: {self.smoothed_pan_offset:.1f}°")
        else:
            # Check for timeout
            if self._check_timeout():
                self.is_active = False
                self.activation_start_time = None
                self.in_cooldown = True
                self.cooldown_start_time = time.time()
                return 0.0

            # Deactivate if offset falls below deactivation threshold
            if abs_offset < self.params.deactivation_threshold:
                self.is_active = False
                self.activation_start_time = None
                logger.debug(f"Waist control deactivated at offset: {self.smoothed_pan_offset:.1f}°")

        # Calculate control value
        if self.is_active:
            # Proportional control normalized to [0, 1]
            # Use offset above deactivation threshold for smoother control
            effective_offset = abs_offset - self.params.deactivation_threshold
            max_effective_offset = 90.0 - self.params.deactivation_threshold

            # Normalize to 0-1 range
            normalized_control = min(effective_offset / max_effective_offset, 1.0)

            # Apply proportional gain and direction
            control_value = normalized_control * (1.0 if self.smoothed_pan_offset > 0 else -1.0)

            # Map to motor throttle with minimum threshold
            self.current_throttle = self._map_throttle(control_value=control_value)
        else:
            self.current_throttle = 0.0

        return self.current_throttle

    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(
            f"Starting WaistNode [activation={self.params.activation_threshold}°, "
            f"deactivation={self.params.deactivation_threshold}°, "
            f"min_throttle={self.params.min_throttle}, "
            f"max_throttle={self.params.max_throttle}, "
            f"timeout={self.params.max_active_duration}s]"
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
                    if self.is_active or abs(throttle) > 0.01:
                        active_time = (
                            time.time() - self.activation_start_time
                            if self.activation_start_time else 0.0
                        )
                        logger.trace(
                            f"Waist: offset={self.smoothed_pan_offset:.1f}°, "
                            f"throttle={throttle:.2f}, active={self.is_active}, "
                            f"active_time={active_time:.1f}s"
                        )

        finally:
            # Ensure motor stops
            self.waist_motor.throttle = 0.0
            await asyncio.sleep(0.5)
            logger.info('WaistNode stopped.')