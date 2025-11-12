import asyncio
import logging
import time

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
    """Simplified parameters for WaistNode."""

    waist_motor_channel: int = Field(default=1, ge=1, le=4)

    # Simple thresholds
    deadzone: float = Field(
        default=15.0,
        ge=5.0,
        le=30.0,
        description="Pan angle offset to ignore (prevent oscillation)"
    )

    # Simple proportional control
    gain: float = Field(
        default=0.015,
        gt=0.0,
        le=0.1,
        description="How aggressively to track (motor_throttle = offset * gain)"
    )

    # Motor throttle range (to overcome static friction)
    min_throttle: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Minimum throttle to overcome static friction"
    )

    max_throttle: float = Field(
        default=1.0,
        ge=0.8,
        le=1.0,
        description="Maximum motor throttle"
    )

    # Safety timeout
    max_active_duration: float = Field(
        default=.5,
        gt=0.0,
        description="Maximum seconds to keep motor running continuously"
    )

    motor_cooldown_duration: float = Field(
        default=1.0,
        gt=0.0,
        description="Seconds to wait after motor timeout before reactivating"
    )


class WaistNode(Node):
    """Simple waist controller that helps the head track targets."""

    params: WaistNodeParams
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    waist_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    head_servo_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)

    # Timeout tracking
    motor_start_time: float | None = Field(default=None)
    # Cooldown tracking - when cooldown ends (absolute time)
    cooldown_end_time: float | None = Field(default=None)

    @classmethod
    def create(
            cls,
            *,
            pubsub: PubSubTopicManager,
            params: WaistNodeParams
    ) -> "WaistNode":
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

        return cls(
            pubsub=pubsub,
            params=params,
            motor_kit=motor_kit,
            waist_motor=waist_motor,
            head_servo_subscription=head_servo_subscription
        )

    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(
            f"Starting WaistNode [deadzone={self.params.deadzone}°, "
            f"gain={self.params.gain}, throttle={self.params.min_throttle:.1f}-{self.params.max_throttle:.1f}, "
            f"timeout={self.params.max_active_duration}s, cooldown={self.params.motor_cooldown_duration}s]"
        )

        try:
            while not self.stop_event.is_set():
                # Short sleep for responsive control
                await asyncio.sleep(0.5)  # 2Hz update rate

                # Get latest head servo state (drain queue to get most recent)
                latest_msg: ServoStateMessage | None = None
                while not self.head_servo_subscription.empty():
                    latest_msg = await self.head_servo_subscription.get()

                if latest_msg is None:
                    # No messages yet, keep motor stopped
                    self.waist_motor.throttle = 0.0
                    continue

                # Calculate and apply throttle (includes timeout and cooldown check)
                throttle = self._calculate_throttle(pan_angle=latest_msg.pan_angle)
                self.waist_motor.throttle = throttle

                # Log only when motor is active or in cooldown
                if abs(throttle) > 0.01:
                    logger.debug(
                        f"Waist: pan={latest_msg.pan_angle:.1f}°, "
                        f"offset={latest_msg.pan_angle - 90:.1f}°, "
                        f"throttle={throttle:.2f}"
                    )
                elif self._is_in_cooldown():
                    remaining_cooldown = self.cooldown_end_time - time.time()
                    logger.debug(
                        f"Waist in cooldown: {remaining_cooldown:.1f}s remaining"
                    )

        except Exception as e:
            logger.error(f"WaistNode error: {e}")
            raise
        finally:
            # Always stop motor on exit
            self.waist_motor.throttle = 0.0
            await asyncio.sleep(0.5)
            logger.info('WaistNode stopped.')

    def _is_in_cooldown(self) -> bool:
        """Check if motor is currently in cooldown period."""
        if self.cooldown_end_time is None:
            return False

        current_time = time.time()
        if current_time < self.cooldown_end_time:
            return True

        # Cooldown has expired, clear it
        self.cooldown_end_time = None
        return False

    def _calculate_throttle(self, *, pan_angle: float) -> float:
        """
        Calculate motor throttle based on pan angle offset from center.

        Args:
            pan_angle: Current pan servo angle (0-180, center=90)

        Returns:
            Motor throttle (0 or min_throttle to max_throttle with sign)
        """
        # Check if in cooldown first
        if self._is_in_cooldown():
            return 0.0

        # Check timeout
        if self.motor_start_time is not None:
            elapsed = time.time() - self.motor_start_time
            if elapsed > self.params.max_active_duration:
                logger.warning(
                    f"Waist motor timeout after {elapsed:.1f}s, entering cooldown for {self.params.motor_cooldown_duration}s")
                # Enter cooldown
                self.cooldown_end_time = time.time() + self.params.motor_cooldown_duration
                self.motor_start_time = None  # Reset for next activation
                return 0.0

        # Calculate offset from center
        offset = pan_angle - 90.0

        # Apply deadzone - motor stops completely
        if abs(offset) < self.params.deadzone:
            self.motor_start_time = None  # Reset timer when stopped
            return 0.0

        # Track when motor starts (but not if we're just coming out of cooldown)
        if self.motor_start_time is None:
            self.motor_start_time = time.time()
            logger.debug(f"Motor starting with offset {offset:.1f}°")

        # Simple proportional control
        raw_throttle = abs(offset) * self.params.gain

        # Scale to min-max throttle range (to overcome static friction)
        throttle_range = self.params.max_throttle - self.params.min_throttle
        scaled_throttle = self.params.min_throttle + (raw_throttle * throttle_range)

        # Clamp to max
        scaled_throttle = min(scaled_throttle, self.params.max_throttle)

        # Apply direction based on offset sign
        final_throttle = -scaled_throttle if offset > 0 else scaled_throttle

        return float(final_throttle)