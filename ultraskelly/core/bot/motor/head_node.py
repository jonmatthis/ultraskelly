import asyncio
import time
from dataclasses import dataclass

import numpy as np
from adafruit_servokit import ServoKit

from pydantic import BaseModel, Field

from ultraskelly.core.bot.__main__bot import logger
from ultraskelly.core.bot.pubsub import PubSub


@dataclass(frozen=True)
class TargetLocationMessage:
    """Generic target location with orientation."""
    x: int | None
    y: int | None
    angle: float | None  # Rotation angle in degrees (0° = vertical up)
    timestamp: float


@dataclass(frozen=True)
class ServoStateMessage:
    """Motor node output."""
    pan_angle: float
    tilt_angle: float
    roll_angle: float
    is_locked_x: bool
    is_locked_y: bool
    is_locked_roll: bool
    timestamp: float


class MotorNodeParams(BaseModel):
    """Parameters for MotorNode."""
    pan_channel: int = Field(default=11, ge=0, le=15)
    tilt_channel: int = Field(default=3, ge=0, le=15)
    roll_channel: int = Field(default=7, ge=0, le=15)
    target_x: int = Field(default=320, ge=0)
    target_y: int = Field(default=240, ge=0)
    offset_x_ratio: float = Field(default=-0.2, ge=-1.0, le=1.0, description="Offset target value in x direction (screen ratio)")
    offset_y_ratio: float = Field(default=0.2, ge=-1.0, le=1.0, description="Offset target value in x direction (screen ratio)")
    gain: float = Field(default=0.05, gt=0.0, le=1.0)
    roll_gain: float = Field(default=0.3, gt=0.0, le=1.0, description="How aggressively to match roll angle")
    roll_smoothing: float = Field(default=0.7, ge=0.0, le=1.0, description="Roll angle smoothing (higher = more smoothing)")
    deadzone: int = Field(default=30, ge=0)
    roll_deadzone: float = Field(default=5.0, ge=0.0, description="Roll angle deadzone in degrees")

    @property
    def x_offset(self) -> int:
        return int(self.offset_x_ratio * self.target_x * 2)
    @property
    def y_offset(self) -> int:
        return int(self.offset_y_ratio * self.target_y * 2)


class MotorNode:
    """Generic servo controller - tracks whatever target is published."""

    def __init__(self, pubsub: PubSub, params: MotorNodeParams) -> None:
        self.pubsub = pubsub
        self.params = params
        self._running = False

        # Initialize servos
        self.kit = ServoKit(channels=16)
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        self.roll_angle = 90.0
        self.kit.servo[params.pan_channel].angle = self.pan_angle
        self.kit.servo[params.tilt_channel].angle = self.tilt_angle
        self.kit.servo[params.roll_channel].angle = self.roll_angle

        # Roll angle smoothing to prevent oscillations
        self.smoothed_target_roll: float | None = None

        self.target_queue = pubsub.target_location.subscribe()

    async def run(self) -> None:
        """Main motor control loop."""
        logger.info(f"Starting MotorNode [gain={self.params.gain}, deadzone={self.params.deadzone}px, roll_gain={self.params.roll_gain}]")
        self._running = True

        try:
            while self._running:
                msg: TargetLocationMessage = await self.target_queue.get()

                if msg.x is None or msg.y is None:
                    await self.pubsub.servo_state.publish(
                        ServoStateMessage(
                            pan_angle=self.pan_angle,
                            tilt_angle=self.tilt_angle,
                            roll_angle=self.roll_angle,
                            is_locked_x=False,
                            is_locked_y=False,
                            is_locked_roll=False,
                            timestamp=time.time()
                        )
                    )
                    continue

                # Calculate errors
                error_x = msg.x - self.params.target_x - self.params.x_offset
                error_y = msg.y - self.params.target_y - self.params.y_offset

                # Check lock status
                is_locked_x = abs(error_x) <= self.params.deadzone
                is_locked_y = abs(error_y) <= self.params.deadzone

                # Update pan/tilt servos if not locked
                if not is_locked_x:
                    self.pan_angle += error_x * self.params.gain
                    self.pan_angle = float(np.clip(self.pan_angle, 0.0, 180.0))
                    self.kit.servo[self.params.pan_channel].angle = self.pan_angle

                if not is_locked_y:
                    self.tilt_angle += error_y * self.params.gain
                    self.tilt_angle = float(np.clip(self.tilt_angle, 0.0, 180.0))
                    self.kit.servo[self.params.tilt_channel].angle = self.tilt_angle

                # Handle roll angle if detected
                is_locked_roll = False
                if msg.angle is not None:
                    # Map rotation angle [-90, 90] to servo angle [0, 180]
                    target_roll = 90.0 + msg.angle

                    # Apply exponential smoothing to prevent oscillations from detection noise
                    if self.smoothed_target_roll is None:
                        self.smoothed_target_roll = target_roll
                    else:
                        # Detect large jumps (likely angle normalization discontinuity at ±90°)
                        jump = abs(target_roll - self.smoothed_target_roll)
                        if jump > 90.0:
                            # Large jump detected - reset smoothing to follow it immediately
                            self.smoothed_target_roll = target_roll
                        else:
                            # Normal update with exponential smoothing
                            self.smoothed_target_roll = (
                                self.params.roll_smoothing * self.smoothed_target_roll +
                                (1.0 - self.params.roll_smoothing) * target_roll
                            )

                    # Calculate roll error using smoothed target
                    roll_error = self.smoothed_target_roll - self.roll_angle

                    # Check if roll is locked
                    is_locked_roll = abs(roll_error) <= self.params.roll_deadzone

                    # Update roll servo if not locked
                    if not is_locked_roll:
                        self.roll_angle += roll_error * self.params.roll_gain
                        self.roll_angle = float(np.clip(self.roll_angle, 0.0, 180.0))
                        self.kit.servo[self.params.roll_channel].angle = self.roll_angle

                await self.pubsub.servo_state.publish(
                    ServoStateMessage(
                        pan_angle=self.pan_angle,
                        tilt_angle=self.tilt_angle,
                        roll_angle=self.roll_angle,
                        is_locked_x=is_locked_x,
                        is_locked_y=is_locked_y,
                        is_locked_roll=is_locked_roll,
                        timestamp=time.time()
                    )
                )
        finally:
            # Center servos
            self.kit.servo[self.params.pan_channel].angle = 90.0
            self.kit.servo[self.params.tilt_channel].angle = 90.0
            self.kit.servo[self.params.roll_channel].angle = 90.0
            await asyncio.sleep(0.5)
            self.kit.servo[self.params.pan_channel].angle = None
            self.kit.servo[self.params.tilt_channel].angle = None
            self.kit.servo[self.params.roll_channel].angle = None
            logger.info("MotorNode stopped")

    async def stop(self) -> None:
        self._running = False
