from time import sleep

from adafruit_motorkit import MotorKit
from gpiozero import RotaryEncoder

# Initialize DC+Stepper HAT at address 0x60 with higher PWM for quieter operation
kit: MotorKit = MotorKit(address=0x60, pwm_frequency=1600)

# Initialize encoder (green=GPIO16, white=GPIO12)
encoder: RotaryEncoder = RotaryEncoder(a=16, b=12, max_steps=1000000)


def move_to_position(
        target_position: int,
        speed: float = 1.0,
        tolerance: int = 5,
        min_throttle: float = .8,
        max_iterations: int = 1000
) -> None:
    """
    Move motor to target encoder position with minimum throttle enforcement.

    Args:
        target_position: Desired encoder position
        speed: Speed multiplier (0.0 to 1.0)
        tolerance: Position error tolerance in encoder steps
        min_throttle: Minimum throttle to overcome static friction
        max_iterations: Maximum control loop iterations before giving up

    Raises:
        RuntimeError: If position not reached within max_iterations
    """
    iterations: int = 0

    while True:
        current_position: int = encoder.steps
        error: int = target_position - current_position

        # Check if we've reached the target
        if abs(error) < tolerance:
            kit.motor1.throttle = 0.0
            print(f"✓ Reached position {current_position}")
            return

        # Check timeout
        iterations += 1
        if iterations > max_iterations:
            kit.motor1.throttle = 0.0
            raise RuntimeError(
                f"Failed to reach target position {target_position} "
                f"(stuck at {current_position} after {iterations} iterations)"
            )

        # Proportional control with minimum throttle enforcement
        throttle_raw: float = error * 0.01 * speed

        # Enforce minimum throttle to overcome static friction
        if abs(throttle_raw) < min_throttle:
            throttle: float = min_throttle if throttle_raw > 0 else -min_throttle
        else:
            throttle = max(-1.0, min(1.0, throttle_raw))

        kit.motor1.throttle = throttle
        print(f"Pos: {current_position:5d} → {target_position:5d} | Error: {error:5d} | Throttle: {throttle:+.2f}")
        sleep(0.05)


def stop_motor() -> None:
    """Immediately stop the motor."""
    kit.motor1.throttle = 0.0


def get_current_position() -> int:
    """Get current encoder position."""
    return encoder.steps


def zero_position() -> None:
    """Set current position as zero."""
    encoder.steps = 0
    print(f"Position zeroed")


# Main demo
if __name__ == "__main__":
    print("DC Motor Position Control Demo")
    print("=" * 50)

    # Zero the encoder at startup
    zero_position()
    print(f"Starting position: {get_current_position()}")

    try:
        # Test sequence
        print("\nMoving to position 500...")
        move_to_position(target_position=500)
        sleep(1)

        print("\nMoving to position 1000...")
        move_to_position(target_position=1000)
        sleep(1)

        print("\nReturning to position 0...")
        move_to_position(target_position=0)
        sleep(1)

        print("\nMoving to position -300...")
        move_to_position(target_position=-300)
        sleep(1)

        print("\nReturning to position 0...")
        move_to_position(target_position=0)

        print("\n✓ Demo complete!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        stop_motor()
    except RuntimeError as e:
        print(f"\n\nERROR: {e}")
        stop_motor()
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")
        stop_motor()
        raise