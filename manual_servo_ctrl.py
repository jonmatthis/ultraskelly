#!/usr/bin/env python3
"""
Simple Manual Servo Control with Camera Preview
"""

import time
import cv2
from picamera2 import Picamera2
from adafruit_servokit import ServoKit


def main() -> None:
    """Main loop - manual servo control."""
    print("="*60)
    print("Manual Servo Control")
    print("="*60)
    
    # Initialize servos
    print("Initializing servos...")
    time.sleep(2)
    kit: ServoKit = ServoKit(channels=16)
    
    # Servo channels
    pitch_channel: int = 3
    yaw_channel: int = 7
    roll_channel: int = 11
    
    # Current angles
    pitch_angle: float = 90.0
    yaw_angle: float = 90.0
    roll_angle: float = 90.0
    
    # Limits
    min_angle: float = 6.0
    max_angle: float = 170.0
    
    # Control step size
    step: float = 5.0
    
    # Center servos
    print("Centering servos...")
    kit.servo[pitch_channel].angle = pitch_angle
    kit.servo[yaw_channel].angle = yaw_angle
    kit.servo[roll_channel].angle = roll_angle
    time.sleep(1)
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    
    print("\n" + "="*60)
    print("Controls:")
    print("  ← → (or J/L) - Yaw left/right")
    print("  ↑ ↓ (or I/K) - Pitch up/down")
    print("  Q / W        - Roll counterclockwise/clockwise")
    print("  R            - Reset to center")
    print("  ESC          - Quit")
    print("\n⚠ IMPORTANT: Click on the camera window to give it focus!")
    print("="*60 + "\n")
    
    debug_keys: bool = True  # Set to False to disable key code printing
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2BGR)
            
            # Add servo info overlay
            text: str = f"Yaw:{yaw_angle:.0f}° Pitch:{pitch_angle:.0f}° Roll:{roll_angle:.0f}°"
            cv2.putText(
                img=frame_bgr,
                text=text,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2
            )
            
            # Show frame
            cv2.imshow("Manual Control", frame_bgr)
            
            # Handle keyboard (longer wait for better key detection)
            key: int = cv2.waitKey(30) & 0xFF
            
            # Debug: print key codes
            if key != 255 and debug_keys:
                print(f"Key pressed: {key}")
            
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):  # Reset
                pitch_angle = 90.0
                yaw_angle = 90.0
                roll_angle = 90.0
                kit.servo[pitch_channel].angle = pitch_angle
                kit.servo[yaw_channel].angle = yaw_angle
                kit.servo[roll_channel].angle = roll_angle
                print("Reset to center")
            # Arrow keys - try multiple possible codes
            elif key == 82 or key == 0:  # Up arrow
                pitch_angle = max(min_angle, min(max_angle, pitch_angle - step))
                kit.servo[pitch_channel].angle = pitch_angle
                print(f"Pitch: {pitch_angle:.0f}°")
            elif key == 84 or key == 1:  # Down arrow
                pitch_angle = max(min_angle, min(max_angle, pitch_angle + step))
                kit.servo[pitch_channel].angle = pitch_angle
                print(f"Pitch: {pitch_angle:.0f}°")
            elif key == 81 or key == 2:  # Left arrow
                yaw_angle = max(min_angle, min(max_angle, yaw_angle + step))
                kit.servo[yaw_channel].angle = yaw_angle
                print(f"Yaw: {yaw_angle:.0f}°")
            elif key == 83 or key == 3:  # Right arrow
                yaw_angle = max(min_angle, min(max_angle, yaw_angle - step))
                kit.servo[yaw_channel].angle = yaw_angle
                print(f"Yaw: {yaw_angle:.0f}°")
            elif key == ord('q') or key == ord('Q'):  # Roll left
                roll_angle = max(min_angle, min(max_angle, roll_angle - step))
                kit.servo[roll_channel].angle = roll_angle
                print(f"Roll: {roll_angle:.0f}°")
            elif key == ord('w') or key == ord('W'):  # Roll right
                roll_angle = max(min_angle, min(max_angle, roll_angle + step))
                kit.servo[roll_channel].angle = roll_angle
                print(f"Roll: {roll_angle:.0f}°")
            # WASD alternatives for arrow keys
            elif key == ord('i') or key == ord('I'):  # Up
                pitch_angle = max(min_angle, min(max_angle, pitch_angle - step))
                kit.servo[pitch_channel].angle = pitch_angle
                print(f"Pitch: {pitch_angle:.0f}°")
            elif key == ord('k') or key == ord('K'):  # Down
                pitch_angle = max(min_angle, min(max_angle, pitch_angle + step))
                kit.servo[pitch_channel].angle = pitch_angle
                print(f"Pitch: {pitch_angle:.0f}°")
            elif key == ord('j') or key == ord('J'):  # Left
                yaw_angle = max(min_angle, min(max_angle, yaw_angle + step))
                kit.servo[yaw_channel].angle = yaw_angle
                print(f"Yaw: {yaw_angle:.0f}°")
            elif key == ord('l') or key == ord('L'):  # Right
                yaw_angle = max(min_angle, min(max_angle, yaw_angle - step))
                kit.servo[yaw_channel].angle = yaw_angle
                print(f"Yaw: {yaw_angle:.0f}°")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Center servos on exit
        kit.servo[pitch_channel].angle = 90.0
        kit.servo[yaw_channel].angle = 90.0
        kit.servo[roll_channel].angle = 90.0
        picam2.stop()
        cv2.destroyAllWindows()
        print("Goodbye!")


if __name__ == "__main__":
    main()