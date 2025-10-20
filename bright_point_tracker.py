#!/usr/bin/env python3
"""
Raspberry Pi Camera with Servo Brightness Tracking
Auto-calibrates servo directions by observing frame movement.
"""

import time
import numpy as np
import cv2
from picamera2 import Picamera2
from adafruit_servokit import ServoKit


class ServoController:
    """Controls servos with automatic direction calibration."""
    
    def __init__(
        self,
        *,
        pitch_channel: int = 3,
        yaw_channel: int = 7,
        roll_channel: int = 11
    ) -> None:
        """
        Initialize servo controller.
        
        Args:
            pitch_channel: Servo channel for pitch (vertical movement)
            yaw_channel: Servo channel for yaw (horizontal movement)
            roll_channel: Servo channel for roll (not used for tracking)
        """
        print("Initializing servo controller...")
        time.sleep(2)
        
        self.kit: ServoKit = ServoKit(channels=16)
        self.pitch_channel: int = pitch_channel
        self.yaw_channel: int = yaw_channel
        self.roll_channel: int = roll_channel
        
        # Current servo angles (start at center)
        self.pitch_angle: float = 90.0
        self.yaw_angle: float = 90.0
        self.roll_angle: float = 90.0
        
        # Servo limits
        self.min_angle: float = 30.0
        self.max_angle: float = 150.0
        
        # Control parameters (will be calibrated)
        self.p_gain: float = 0.1
        self.deadzone: float = 30.0
        
        # Calibration results - polarity determines direction
        self.yaw_polarity: float = -1.0  # Will be calibrated
        self.pitch_polarity: float = 1.0  # Will be calibrated
        self.calibrated: bool = False
        
        # Center servos
        self.center_servos()
        print("Servo controller initialized!")
    
    def center_servos(self) -> None:
        """Move all servos to center position."""
        print("Centering servos...")
        self.pitch_angle = 90.0
        self.yaw_angle = 90.0
        self.roll_angle = 90.0
        
        self.kit.servo[self.pitch_channel].angle = self.pitch_angle
        self.kit.servo[self.yaw_channel].angle = self.yaw_angle
        self.kit.servo[self.roll_channel].angle = self.roll_angle
        time.sleep(1)
    
    def clamp_angle(self, *, angle: float) -> float:
        """Clamp angle to safe servo range."""
        return max(self.min_angle, min(self.max_angle, angle))
    
    def move_servo(self, *, channel: int, delta: float) -> None:
        """
        Move a specific servo by a delta amount.
        
        Args:
            channel: Servo channel to move
            delta: Amount to move (degrees)
        """
        if channel == self.pitch_channel:
            self.pitch_angle = self.clamp_angle(angle=self.pitch_angle + delta)
            self.kit.servo[channel].angle = self.pitch_angle
        elif channel == self.yaw_channel:
            self.yaw_angle = self.clamp_angle(angle=self.yaw_angle + delta)
            self.kit.servo[channel].angle = self.yaw_angle
    
    def calibrate(self, *, get_target_position_func) -> None:
        """
        Auto-calibrate servo directions by making test movements.
        
        Args:
            get_target_position_func: Function that returns (x, y) of target
        """
        print("\n" + "="*60)
        print("CALIBRATING SERVO DIRECTIONS")
        print("="*60)
        print("Please ensure a bright point is visible in the frame...")
        time.sleep(2)
        
        # Center servos first
        self.center_servos()
        time.sleep(1)
        
        calibration_delta: float = 15.0  # Degrees to move for test
        
        # Calibrate YAW (horizontal)
        print("\nCalibrating YAW (horizontal) servo...")
        
        # Get initial position
        initial_x, initial_y = get_target_position_func()
        print(f"  Initial target position: ({initial_x}, {initial_y})")
        
        # Move yaw servo positive direction
        self.move_servo(channel=self.yaw_channel, delta=calibration_delta)
        time.sleep(0.5)
        
        # Get new position
        new_x, new_y = get_target_position_func()
        print(f"  After +{calibration_delta}° yaw: ({new_x}, {new_y})")
        
        # Determine polarity
        # If servo increases and target moves right (x increases), polarity is negative
        # (because we want to decrease servo angle to move target left toward center)
        x_movement: float = new_x - initial_x
        if abs(x_movement) > 10:  # Significant movement detected
            # Positive servo movement causing positive x movement means inverse relationship
            self.yaw_polarity = -1.0 if x_movement > 0 else 1.0
            print(f"  X movement: {x_movement:.1f} pixels")
            print(f"  ✓ Yaw polarity set to: {self.yaw_polarity}")
        else:
            print(f"  ⚠ Minimal movement detected, using default polarity")
            self.yaw_polarity = -1.0
        
        # Return to center
        self.center_servos()
        time.sleep(1)
        
        # Calibrate PITCH (vertical)
        print("\nCalibrating PITCH (vertical) servo...")
        
        # Get initial position
        initial_x, initial_y = get_target_position_func()
        print(f"  Initial target position: ({initial_x}, {initial_y})")
        
        # Move pitch servo positive direction
        self.move_servo(channel=self.pitch_channel, delta=calibration_delta)
        time.sleep(0.5)
        
        # Get new position
        new_x, new_y = get_target_position_func()
        print(f"  After +{calibration_delta}° pitch: ({new_x}, {new_y})")
        
        # Determine polarity
        y_movement: float = new_y - initial_y
        if abs(y_movement) > 10:  # Significant movement detected
            # Positive servo movement causing positive y movement means same relationship
            self.pitch_polarity = 1.0 if y_movement > 0 else -1.0
            print(f"  Y movement: {y_movement:.1f} pixels")
            print(f"  ✓ Pitch polarity set to: {self.pitch_polarity}")
        else:
            print(f"  ⚠ Minimal movement detected, using default polarity")
            self.pitch_polarity = 1.0
        
        # Return to center
        self.center_servos()
        
        self.calibrated = True
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print(f"Yaw polarity: {self.yaw_polarity}")
        print(f"Pitch polarity: {self.pitch_polarity}")
        print("="*60 + "\n")
        time.sleep(1)
    
    def track_target(
        self,
        *,
        target_x: int,
        target_y: int,
        frame_width: int,
        frame_height: int
    ) -> None:
        """
        Adjust servos to center the target in frame.
        
        Args:
            target_x: X coordinate of target
            target_y: Y coordinate of target
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        if not self.calibrated:
            return
        
        # Calculate center of frame
        center_x: float = frame_width / 2.0
        center_y: float = frame_height / 2.0
        
        # Calculate error
        error_x: float = target_x - center_x
        error_y: float = target_y - center_y
        
        # Apply deadzone
        if abs(error_x) < self.deadzone and abs(error_y) < self.deadzone:
            return
        
        # Calculate adjustments with calibrated polarities
        yaw_adjustment: float = self.yaw_polarity * error_x * self.p_gain
        pitch_adjustment: float = self.pitch_polarity * error_y * self.p_gain
        
        # Update angles
        self.yaw_angle = self.clamp_angle(angle=self.yaw_angle + yaw_adjustment)
        self.pitch_angle = self.clamp_angle(angle=self.pitch_angle + pitch_adjustment)
        
        # Apply to servos
        self.kit.servo[self.yaw_channel].angle = self.yaw_angle
        self.kit.servo[self.pitch_channel].angle = self.pitch_angle


def find_brightest_point(*, frame: np.ndarray) -> tuple[int, int, float]:
    """
    Find the brightest point in the frame.
    
    Args:
        frame: BGR image array
        
    Returns:
        Tuple of (x, y, brightness_value)
    """
    # Convert to grayscale
    gray: np.ndarray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    
    # Find the location of maximum brightness
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=gray)
    
    return max_loc[0], max_loc[1], max_val


def main() -> None:
    """Main loop - track brightest point in frame."""
    print("="*60)
    print("Self-Calibrating Brightness Tracking System")
    print("="*60)
    
    # Initialize servo controller
    servo_controller = ServoController(
        pitch_channel=3,
        yaw_channel=7,
        roll_channel=11
    )
    
    # Initialize camera
    print("\nInitializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Let camera warm up
    
    # Function to get current target position
    def get_target_position() -> tuple[int, int]:
        frame: np.ndarray = picam2.capture_array()
        frame_bgr: np.ndarray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2BGR)
        x, y, _ = find_brightest_point(frame=frame_bgr)
        return x, y
    
    # Run calibration
    servo_controller.calibrate(get_target_position_func=get_target_position)
    
    print("\n" + "="*60)
    print("TRACKING ACTIVE!")
    print("- Tracks brightest point in frame")
    print("- Red circle = tracked point")
    print("- Blue crosshair = frame center")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    frame_count: int = 0
    
    try:
        while True:
            # Capture frame
            frame: np.ndarray = picam2.capture_array()
            
            # Convert from RGB to BGR for OpenCV
            frame_bgr: np.ndarray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2BGR)
            
            height, width = frame_bgr.shape[:2]
            
            # Find brightest point
            bright_x, bright_y, brightness = find_brightest_point(frame=frame_bgr)
            
            # Track with servos
            servo_controller.track_target(
                target_x=bright_x,
                target_y=bright_y,
                frame_width=width,
                frame_height=height
            )
            
            # Draw center crosshair (blue)
            center_x: int = width // 2
            center_y: int = height // 2
            cv2.drawMarker(
                img=frame_bgr,
                position=(center_x, center_y),
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=30,
                thickness=2
            )
            
            # Draw brightest point (red circle)
            cv2.circle(
                img=frame_bgr,
                center=(bright_x, bright_y),
                radius=10,
                color=(0, 0, 255),
                thickness=3
            )
            
            # Draw connecting line
            cv2.line(
                img=frame_bgr,
                pt1=(center_x, center_y),
                pt2=(bright_x, bright_y),
                color=(0, 255, 0),
                thickness=1
            )
            
            # Add info text
            info_text: str = f"Brightness: {brightness:.0f} | Pos: ({bright_x}, {bright_y})"
            cv2.putText(
                img=frame_bgr,
                text=info_text,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=2
            )
            
            calib_text: str = f"Yaw: {servo_controller.yaw_polarity:+.0f} | Pitch: {servo_controller.pitch_polarity:+.0f}"
            cv2.putText(
                img=frame_bgr,
                text=calib_text,
                org=(10, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=2
            )
            
            # Show frame
            cv2.imshow("Brightness Tracking", frame_bgr)
            
            # Print status periodically
            if frame_count % 30 == 0:
                error_x: float = bright_x - center_x
                error_y: float = bright_y - center_y
                print(f"Frame {frame_count}: Brightness={brightness:.0f}, "
                      f"Error=({error_x:.0f}, {error_y:.0f}), "
                      f"Servos=(Y:{servo_controller.yaw_angle:.1f}°, P:{servo_controller.pitch_angle:.1f}°)")
            
            frame_count += 1
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.033)  # ~30 fps
            
    except KeyboardInterrupt:
        print("\n\nStopping tracking system...")
    
    finally:
        # Clean up
        servo_controller.center_servos()
        picam2.stop()
        cv2.destroyAllWindows()
        print("System stopped. Goodbye!")


if __name__ == "__main__":
    main()