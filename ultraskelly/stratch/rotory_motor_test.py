from adafruit_motorkit import MotorKit
from time import sleep


def test_motor_configuration(
        motor_number: int = 1,
        pwm_frequencies: list[int] = [1600, 5000, 10000, 15000, 20000, 25000],
        throttle_values: list[float] = [0.3, 0.5, 0.7, 1.0],
        test_duration: float = 2.0
) -> None:
    """Sweep through PWM frequencies and throttle values to find what works."""

    print("=" * 60)
    print("MOTOR DIAGNOSTIC SWEEP")
    print("=" * 60)
    print("\nWatch the motor and listen for changes!")
    print("Note which combination makes it actually SPIN (not just hum)")
    print()

    for pwm_freq in pwm_frequencies:
        print(f"\n{'=' * 60}")
        print(f"Testing PWM Frequency: {pwm_freq} Hz")
        print(f"{'=' * 60}")

        try:
            kit: MotorKit = MotorKit(pwm_frequency=pwm_freq)
            motor = kit.motor1 if motor_number == 1 else kit.motor2

            for throttle in throttle_values:
                print(f"\n  → Throttle: {throttle:.1f} ({int(throttle * 100)}%)")
                print(f"     Testing forward...", end=" ", flush=True)

                motor.throttle = throttle
                sleep(test_duration)
                motor.throttle = 0.0

                print("STOP", end=" ")
                sleep(0.5)

                print("→ Reverse...", end=" ", flush=True)
                motor.throttle = -throttle
                sleep(test_duration)
                motor.throttle = 0.0

                print("STOP")
                sleep(1.0)

        except Exception as e:
            print(f"\n  ERROR at {pwm_freq}Hz: {e}")
            continue

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE!")
    print("=" * 60)
    print("\nDid the motor spin at any combination?")
    print("If NO: Check external power supply to Motor HAT!")
    print("If YES: Note which PWM frequency worked best")


if __name__ == "__main__":
    # Run the sweep
    test_motor_configuration()