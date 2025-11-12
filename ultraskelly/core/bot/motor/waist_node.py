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

    # Hysteresis thresholds
    activation_threshold: float = Field(default=15.0, ge=0.0, le=90.0,
                                        description="Pan angle offset to start waist movement")
    deactivation_threshold: float = Field(default=5.0, ge=0.0, le=90.0,
                                          description="Pan angle offset to stop waist movement")

    # Deceleration zone for overdamping
    deceleration_zone: float = Field(default=25.0, ge=10.0, le=45.0,
                                     description="Degrees from target to start decelerating")

    # Motor control parameters
    min_throttle: float = Field(default=0.7, ge=0.5, le=1.0,
                                description="Minimum throttle to overcome static friction")
    max_throttle: float = Field(default=1.0, ge=0.7, le=1.0,
                                description="Maximum throttle limit")

    # PD control gains (reduced P, added D for overdamping)
    proportional_gain: float = Field(default=0.008, gt=0.0, le=0.1,
                                     description="P-gain for waist control (reduced for overdamping)")
    derivative_gain: float = Field(default=0.15, ge=0.0, le=1.0,
                                   description="D-gain for damping velocity")

    # Safety parameters
    max_active_duration: float = Field(default=4.0, gt=0.0,
                                       description="Maximum time to keep motor active before timeout")
    cooldown_duration: float = Field(default=2.0, gt=0.0,
                                     description="Cooldown period after timeout")

    # Smoothing parameters
    angle_smoothing_window: int = Field(default=15, ge=1, le=50,
                                        description="Window size for angle smoothing")
    velocity_smoothing_window: int = Field(default=5, ge=1, le=20,
                                           description="Window size for velocity estimation")


class WaistNode(Node):
    """Overdamped waist controller with anti-overshoot protection."""

    params: WaistNodeParams
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    waist_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    head_servo_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)

    # State tracking
    is_active: bool = Field(default=False)
    activation_start_time: float | None = Field(default=None)
    in_cooldown: bool = Field(default=False)
    cooldown_start_time: float | None = Field(default=None)

    # Control values
    current_throttle: float = Field(default=0.0)
    pan_angle_buffer: deque[float] = Field(default_factory=lambda: deque(maxlen=15))
    velocity_buffer: deque[float] = Field(default_factory=lambda: deque(maxlen=5))
    smoothed_pan_offset: float = Field(default=0.0)
    estimated_velocity: float = Field(default=0.0)
    last_update_time: float = Field(default=0.0)

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

        # Create instance with proper buffer sizes
        instance = cls(
            pubsub=pubsub,
            params=params,
            motor_kit=motor_kit,
            waist_motor=waist_motor,
            head_servo_subscription=head_servo_subscription
        )
        instance.pan_angle_buffer = deque(maxlen=params.angle_smoothing_window)
        instance.velocity_buffer = deque(maxlen=params.velocity_smoothing_window)
        instance.last_update_time = time.time()

        return instance

    def _estimate_velocity(self, *, new_offset: float) -> float:
        """Estimate angular velocity for derivative damping."""
        current_time = time.time()
        dt = current_time - self.last_update_time

        if dt > 0.001 and len(self.pan_angle_buffer) > 0:
            # Calculate instantaneous velocity
            instant_velocity = (new_offset - self.smoothed_pan_offset) / dt
            self.velocity_buffer.append(instant_velocity)

            # Smooth velocity estimate
            if len(self.velocity_buffer) > 0:
                self.estimated_velocity = float(np.mean(self.velocity_buffer))

        self.last_update_time = current_time
        return self.estimated_velocity

    def _calculate_overdamped_throttle(self, *, offset: float, velocity: float) -> float:
        """Calculate throttle with overdamping characteristics."""
        abs_offset = abs(offset)

        # Determine if we're in deceleration zone
        if abs_offset < self.params.deceleration_zone:
            # Progressive deceleration as we approach target
            decel_factor = abs_offset / self.params.deceleration_zone
            # Cubic easing for smoother deceleration
            decel_factor = decel_factor ** 2
        else:
            decel_factor = 1.0

        # Check if velocity is opposing offset (good - we're slowing down)
        # or same direction (bad - we're accelerating toward overshoot)
        velocity_opposes = (offset * velocity) < 0

        # Calculate base control
        # P term - proportional to error with deceleration
        p_term = self.params.proportional_gain * offset * decel_factor

        # D term - derivative damping (opposes velocity)
        d_term = -self.params.derivative_gain * velocity

        # If we're approaching target fast, increase damping
        if not velocity_opposes and abs(velocity) > 5.0:
            d_term *= 2.0  # Double damping when moving fast toward target

        # Combined PD control
        control_value = p_term + d_term

        # Limit control value
        control_value = np.clip(control_value, -1.0, 1.0)

        return float(control_value)

    def _map_throttle(self, *, control_value: float) -> float:
        """Map control value to motor throttle with minimum threshold."""
        if abs(control_value) < 0.001:
            return 0.0

        # For overdamping, use a lower minimum throttle in decel zone
        abs_offset = abs(self.smoothed_pan_offset)
        if abs_offset < self.params.deceleration_zone:
            # Allow lower throttle when close to target for fine control
            effective_min = self.params.min_throttle * 0.85
        else:
            effective_min = self.params.min_throttle

        throttle_range = self.params.max_throttle - effective_min
        sign = 1.0 if control_value > 0 else -1.0

        # Scale control value to throttle range
        scaled_throttle = effective_min + abs(control_value) * throttle_range

        # Apply sign and clamp
        final_throttle = sign * scaled_throttle
        return float(np.clip(final_throttle, -self.params.max_throttle, self.params.max_throttle))

    def _check_timeout(self) -> bool:
        """Check if motor has been active too long."""
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
        """Calculate overdamped waist control."""
        # Check cooldown first
        if self._check_cooldown():
            return 0.0

        # Calculate offset from center
        pan_offset = pan_angle - 90.0

        # Estimate velocity before updating buffer
        velocity = self._estimate_velocity(new_offset=pan_offset)

        # Add to smoothing buffer
        self.pan_angle_buffer.append(pan_offset)

        # Calculate heavily smoothed offset for overdamping
        if len(self.pan_angle_buffer) > 0:
            # Use weighted average favoring recent values
            weights = np.linspace(0.5, 1.0, len(self.pan_angle_buffer))
            self.smoothed_pan_offset = float(np.average(self.pan_angle_buffer, weights=weights))
        else:
            self.smoothed_pan_offset = pan_offset

        abs_offset = abs(self.smoothed_pan_offset)

        # Hysteresis with velocity consideration
        if not self.is_active:
            # Only activate if offset is large AND we're not moving away
            if abs_offset > self.params.activation_threshold and abs(velocity) < 20.0:
                self.is_active = True
                self.activation_start_time = time.time()
                self.velocity_buffer.clear()  # Reset velocity on activation
                logger.debug(f"Waist activated: offset={self.smoothed_pan_offset:.1f}°")
        else:
            # Check timeout
            if self._check_timeout():
                self.is_active = False
                self.activation_start_time = None
                self.in_cooldown = True
                self.cooldown_start_time = time.time()
                return 0.0

            # Deactivate when close AND slow
            if abs_offset < self.params.deactivation_threshold and abs(velocity) < 5.0:
                self.is_active = False
                self.activation_start_time = None
                logger.debug(f"Waist deactivated: offset={self.smoothed_pan_offset:.1f}°, vel={velocity:.1f}")

        # Calculate control
        if self.is_active:
            # Get overdamped control value
            control = self._calculate_overdamped_throttle(
                offset=self.smoothed_pan_offset,
                velocity=velocity
            )

            # Map to motor throttle
            self.current_throttle = self._map_throttle(control_value=control)
        else:
            self.current_throttle = 0.0

        return self.current_throttle

    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(
            f"Starting WaistNode OVERDAMPED [act={self.params.activation_threshold}°, "
            f"deact={self.params.deactivation_threshold}°, "
            f"decel_zone={self.params.deceleration_zone}°, "
            f"P={self.params.proportional_gain}, D={self.params.derivative_gain}]"
        )

        try:
            latest_pan_angle: float | None = None

            while not self.stop_event.is_set():
                await asyncio.sleep(0.01)  # 100Hz update

                # Get latest head servo state
                while not self.head_servo_subscription.empty():
                    msg: ServoStateMessage = await self.head_servo_subscription.get()
                    latest_pan_angle = msg.pan_angle

                # Calculate and apply control
                if latest_pan_angle is not None:
                    throttle = self._calculate_control(pan_angle=latest_pan_angle)
                    self.waist_motor.throttle = throttle

                    # Debug logging
                    if self.is_active or abs(throttle) > 0.01:
                        logger.trace(
                            f"Waist: offset={self.smoothed_pan_offset:.1f}°, "
                            f"vel={self.estimated_velocity:.1f}°/s, "
                            f"throttle={throttle:.2f}"
                        )

        finally:
            self.waist_motor.throttle = 0.0
            await asyncio.sleep(0.5)
            logger.info('WaistNode stopped.')