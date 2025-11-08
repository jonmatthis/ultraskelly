from time import sleep

<<<<<<< HEAD
pwm_frequencies: list[int] = [1600]#, 5000, 10000, 15000, 20000, 25000]
=======
from adafruit_motorkit import MotorKit

pwm_frequencies: list[int] = [1600, 5000, 10000, 15000, 20000, 25000]
>>>>>>> 2b150cea738de9d6a311fdeb6700e5f945f1380c

for freq in pwm_frequencies:
    print(f"\n{'=' * 50}")
    print(f"Testing PWM frequency: {freq} Hz at address 0x60")
    print(f"{'=' * 50}")

    try:
        kit: MotorKit = MotorKit(address=0x60, pwm_frequency=freq)

        print("Motor at 70% forward...")
        kit.motor1.throttle = 1.0
        sleep(8 m)
        kit.motor1.throttle = -1.0
        sleep(8)
    
        kit.motor1.throttle = 0.0

        print(f"✓ {freq}Hz worked! How did it sound/perform?")
        sleep(1)

    except Exception as e:
        print(f"✗ {freq}Hz failed: {e}")

print("\nWhich frequency worked best?")