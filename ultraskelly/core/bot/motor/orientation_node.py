import asyncio
import logging
import time
from enum import Enum

import numpy as np
from adafruit_motor.motor import DCMotor
from adafruit_motorkit import MotorKit
from adafruit_servokit import ServoKit
from pydantic import Field, SkipValidation

from ultraskelly.core.bot.base_abcs import Node, NodeParams
from ultraskelly.core.pubsub.bot_topics import (
    ServoStateMessage,
    ServoStateTopic,
    TargetLocationMessage,
    TargetLocationTopic,
)
from ultraskelly.core.pubsub.pubsub_abcs import TopicSubscriptionQueue
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager

logger = logging.getLogger(__name__)


class WaistControlMode(Enum):
    """Waist motor control modes."""
    IDLE = "idle"  # Not moving
    CENTERING = "centering"  # Trying to center the head
    ASSISTING = "assisting"  # Helping head reach target


class OrientationNodeParams(NodeParams):
    """Parameters for OrientationNode."""
    
    # Servo channels
    pan_channel: int = Field(default=11, ge=0, le=15)
    tilt_channel: int = Field(default=3, ge=0, le=15)
    
    # Waist motor channel
    waist_motor_channel: int = Field(default=1, ge=1, le=4)
    
    # Target tracking
    target_x: int = Field(default=320, ge=0)
    target_y: int = Field(default=240, ge=0)
    offset_x_ratio: float = Field(default=0., ge=-1.0, le=1.0)
    offset_y_ratio: float = Field(default=0., ge=-1.0, le=1.0)
    
    # Head servo control
    head_gain: float = Field(default=0.05, gt=0.0, le=1.0)
    head_deadzone: int = Field(default=100, ge=0)
    
    # Waist motor control
    waist_min_throttle: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Minimum throttle to overcome static friction"
    )
    waist_max_throttle: float = Field(
        default=0.8,
        ge=0.7,
        le=1.0,
        description="Maximum waist motor throttle"
    )
    waist_gain: float = Field(
        default=0.008,
        gt=0.0,
        le=0.05,
        description="Waist motor proportional gain"
    )
    
    # Pan angle thresholds for waist engagement
    pan_center_zone: float = Field(
        default=15.0,
        ge=5.0,
        le=30.0,
        description="Pan angle from center (90°) within which waist doesn't need to center"
    )
    pan_assist_threshold: float = Field(
        default=45.0,
        ge=30.0,
        le=60.0,
        description="Pan angle from center beyond which waist assists"
    )
    pan_limit_threshold: float = Field(
        default=60.0,
        ge=45.0,
        le=80.0,
        description="Pan angle from center where waist must help (near servo limits)"
    )
    
    # Safety and timing
    waist_max_active_duration: float = Field(
        default=0.5,
        gt=0.0,
        description="Maximum seconds to keep waist motor running continuously"
    )
    waist_cooldown_duration: float = Field(
        default=1.0,
        gt=0.0,
        description="Seconds to wait after waist timeout before reactivating"
    )
    
    @property
    def x_offset(self) -> int:
        return int(self.offset_x_ratio * self.target_x * 2)
    
    @property
    def y_offset(self) -> int:
        return int(self.offset_y_ratio * self.target_y * 2)


class OrientationNode(Node):
    """Combined head and waist controller for maximum tracking range."""
    
    params: OrientationNodeParams
    
    # Hardware
    servo_kit: SkipValidation[ServoKit] = Field(default=None, exclude=True)
    motor_kit: SkipValidation[MotorKit] = Field(default=None, exclude=True)
    waist_motor: SkipValidation[DCMotor] = Field(default=None, exclude=True)
    
    # Subscriptions
    target_subscription: SkipValidation[TopicSubscriptionQueue] = Field(default=None, exclude=True)
    
    # Head servo state
    pan_angle: float = Field(default=90.0)
    tilt_angle: float = Field(default=90.0)
    
    # Waist motor state
    waist_mode: WaistControlMode = Field(default=WaistControlMode.IDLE)
    waist_motor_start_time: float | None = Field(default=None)
    waist_cooldown_end_time: float | None = Field(default=None)
    
    # Target tracking state
    last_target_x: int | None = Field(default=None)
    last_target_y: int | None = Field(default=None)
    
    @classmethod
    def create(
        cls,
        *,
        pubsub: PubSubTopicManager,
        params: OrientationNodeParams
    ) -> "OrientationNode":
        """Factory method to create and initialize OrientationNode."""
        node = cls(pubsub=pubsub, params=params)
        
        # Initialize servo kit
        node.servo_kit = ServoKit(channels=16)
        node.servo_kit.servo[params.pan_channel].angle = node.pan_angle
        node.servo_kit.servo[params.tilt_channel].angle = node.tilt_angle
        
        # Initialize motor kit
        node.motor_kit = MotorKit(address=0x60, pwm_frequency=1600)
        
        # Select waist motor channel
        match params.waist_motor_channel:
            case 1:
                node.waist_motor = node.motor_kit.motor1
            case 2:
                node.waist_motor = node.motor_kit.motor2
            case 3:
                node.waist_motor = node.motor_kit.motor3
            case 4:
                node.waist_motor = node.motor_kit.motor4
            case _:
                raise ValueError(f"Invalid waist motor channel: {params.waist_motor_channel}")
        
        if node.waist_motor is None:
            raise ValueError(f"Could not initialize waist motor on channel {params.waist_motor_channel}")
        
        # Get subscription
        node.target_subscription = pubsub.get_subscription(TargetLocationTopic)
        
        return node
    
    def _update_head_servos(
        self,
        *,
        target_x: int,
        target_y: int
    ) -> tuple[bool, bool]:
        """
        Update head servo positions to track target.
        
        Returns:
            Tuple of (is_locked_x, is_locked_y)
        """
        # Calculate errors
        error_x = target_x - self.params.target_x - self.params.x_offset
        error_y = target_y - self.params.target_y - self.params.y_offset
        
        # Check lock status
        is_locked_x = abs(error_x) <= self.params.head_deadzone
        is_locked_y = abs(error_y) <= self.params.head_deadzone
        
        # Update pan servo
        if not is_locked_x:
            self.pan_angle += error_x * self.params.head_gain
            self.pan_angle = float(np.clip(self.pan_angle, 0.0, 180.0))
            self.servo_kit.servo[self.params.pan_channel].angle = self.pan_angle
        
        # Update tilt servo
        if not is_locked_y:
            self.tilt_angle += error_y * self.params.head_gain
            self.tilt_angle = float(np.clip(self.tilt_angle, 0.0, 180.0))
            self.servo_kit.servo[self.params.tilt_channel].angle = self.tilt_angle
        
        return is_locked_x, is_locked_y
    
    def _is_waist_in_cooldown(self) -> bool:
        """Check if waist motor is in cooldown period."""
        if self.waist_cooldown_end_time is None:
            return False
        
        current_time = time.time()
        if current_time < self.waist_cooldown_end_time:
            return True
        
        # Cooldown expired
        self.waist_cooldown_end_time = None
        return False
    
    def _check_waist_timeout(self) -> bool:
        """Check if waist motor has timed out."""
        if self.waist_motor_start_time is None:
            return False
        
        elapsed = time.time() - self.waist_motor_start_time
        if elapsed > self.params.waist_max_active_duration:
            logger.warning(
                f"Waist motor timeout after {elapsed:.1f}s, "
                f"entering cooldown for {self.params.waist_cooldown_duration}s"
            )
            # Enter cooldown
            self.waist_cooldown_end_time = time.time() + self.params.waist_cooldown_duration
            self.waist_motor_start_time = None
            self.waist_mode = WaistControlMode.IDLE
            return True
        
        return False
    
    def _determine_waist_mode(
        self,
        *,
        is_locked_x: bool
    ) -> WaistControlMode:
        """
        Determine what mode the waist should be in based on head position and target lock.
        """
        pan_offset = abs(self.pan_angle - 90.0)
        
        # If head is locked on target in X
        if is_locked_x:
            # If pan is far from center, try to center it
            if pan_offset > self.params.pan_center_zone:
                return WaistControlMode.CENTERING
            else:
                return WaistControlMode.IDLE
        
        # Head is not locked on target
        else:
            # If pan is near limits, waist must help
            if pan_offset > self.params.pan_limit_threshold:
                return WaistControlMode.ASSISTING
            # If pan is getting far from center, waist should help
            elif pan_offset > self.params.pan_assist_threshold:
                return WaistControlMode.ASSISTING
            else:
                return WaistControlMode.IDLE
    
    def _calculate_waist_throttle(
        self,
        *,
        mode: WaistControlMode,
        target_x: int | None
    ) -> float:
        """
        Calculate waist motor throttle based on mode and head position.
        """
        # Check cooldown first
        if self._is_waist_in_cooldown():
            return 0.0
        
        # Check timeout
        if self._check_waist_timeout():
            return 0.0
        
        # No movement in idle mode
        if mode == WaistControlMode.IDLE:
            self.waist_motor_start_time = None
            return 0.0
        
        # Track motor start time
        if self.waist_motor_start_time is None:
            self.waist_motor_start_time = time.time()
            logger.debug(f"Waist motor starting in {mode.value} mode")
        
        # Calculate base throttle based on mode
        if mode == WaistControlMode.CENTERING:
            # Try to center the head at 90°
            pan_offset = self.pan_angle - 90.0
            
            # Small deadzone to prevent oscillation when centered
            if abs(pan_offset) < 15.0:
                self.waist_motor_start_time = None
                return 0.0
            
            # Proportional control to center
            raw_throttle = abs(pan_offset) * self.params.waist_gain
            
        elif mode == WaistControlMode.ASSISTING:
            # Help head reach target
            if target_x is None:
                return 0.0
            
            # Use both pan offset and target error for aggressive assistance
            pan_offset = self.pan_angle - 90.0
            target_error = target_x - self.params.target_x - self.params.x_offset
            
            # Combine signals: pan offset tells us which way we're leaning,
            # target error tells us which way we need to go
            if np.sign(pan_offset) == np.sign(target_error):
                # Head and target are in same direction, use stronger signal
                raw_throttle = max(abs(pan_offset) * self.params.waist_gain * 1.5,
                                 abs(target_error) * 0.001)
            else:
                # Conflicting signals, be conservative
                raw_throttle = abs(pan_offset) * self.params.waist_gain * 0.5
            
            # Determine direction from pan offset
            pan_offset = np.sign(pan_offset) * raw_throttle
            
        else:
            return 0.0
        
        # Scale throttle to min-max range
        throttle_range = self.params.waist_max_throttle - self.params.waist_min_throttle
        scaled_throttle = self.params.waist_min_throttle + (raw_throttle * throttle_range)
        scaled_throttle = min(scaled_throttle, self.params.waist_max_throttle)
        
        # Apply direction
        if mode == WaistControlMode.CENTERING:
            # Move opposite to pan offset to center
            # final_throttle = scaled_throttle if pan_offset > 0 else -scaled_throttle
            final_throttle = 0 # turn off centering for now
        else:  # ASSISTING
            # Move in direction of pan offset
            final_throttle = -scaled_throttle if pan_offset > 0 else scaled_throttle
        
        return float(final_throttle)
    
    async def run(self) -> None:
        """Main control loop."""
        logger.info(
            f"Starting OrientationNode [head_gain={self.params.head_gain}, "
            f"waist_gain={self.params.waist_gain}, "
            f"waist_throttle={self.params.waist_min_throttle:.1f}-{self.params.waist_max_throttle:.1f}]"
        )
        
        try:
            while not self.stop_event.is_set():
                # Get latest target (drain queue to get most recent)
                latest_msg: TargetLocationMessage | None = None
                while not self.target_subscription.empty():
                    latest_msg = await self.target_subscription.get()
                
                # Process target if we have one
                if latest_msg and latest_msg.x is not None and latest_msg.y is not None:
                    # Update head servos first
                    is_locked_x, is_locked_y = self._update_head_servos(
                        target_x=latest_msg.x,
                        target_y=latest_msg.y
                    )
                    
                    # Determine waist mode
                    waist_mode = self._determine_waist_mode(is_locked_x=is_locked_x)
                    
                    # Calculate and apply waist throttle
                    waist_throttle = self._calculate_waist_throttle(
                        mode=waist_mode,
                        target_x=latest_msg.x
                    )
                    self.waist_motor.throttle = waist_throttle
                    
                    # Store state
                    self.waist_mode = waist_mode
                    self.last_target_x = latest_msg.x
                    self.last_target_y = latest_msg.y
                    
                    # Log status when active
                    if abs(waist_throttle) > 0.01 or waist_mode != WaistControlMode.IDLE:
                        logger.debug(
                            f"Orientation: pan={self.pan_angle:.1f}°, "
                            f"locked_x={is_locked_x}, waist_mode={waist_mode.value}, "
                            f"throttle={waist_throttle:.2f}"
                        )
                    
                    # Publish servo state
                    await self.pubsub.publish(
                        ServoStateTopic,
                        ServoStateMessage(
                            pan_angle=self.pan_angle,
                            tilt_angle=self.tilt_angle,
                            roll_angle=90.0,  # Not used in this node
                            is_locked_x=is_locked_x,
                            is_locked_y=is_locked_y,
                            is_locked_roll=False
                        )
                    )
                
                else:
                    # No target, stop waist
                    self.waist_motor.throttle = 0.0
                    self.waist_mode = WaistControlMode.IDLE
                    self.waist_motor_start_time = None
                
                # Short sleep for responsive control
                await asyncio.sleep(0.01)  # 100Hz update rate
        
        except Exception as e:
            logger.error(f"OrientationNode error: {e}")
            raise
        
        finally:
            # Always stop motors and center servos on exit
            self.waist_motor.throttle = 0.0
            self.servo_kit.servo[self.params.pan_channel].angle = 90.0
            self.servo_kit.servo[self.params.tilt_channel].angle = 90.0
            await asyncio.sleep(0.5)
            logger.info("OrientationNode stopped")