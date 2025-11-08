from time import sleep

from adafruit_motorkit import MotorKit
from gpiozero import DigitalInputDevice

# Connect green to GPIO 16, white to GPIO 12 (like you have now)
print("Test: Are encoder signals already present?")
print("=" * 50)

# Just read the pins as digital inputs
green: DigitalInputDevice = DigitalInputDevice(16, pull_up=True)
white: DigitalInputDevice = DigitalInputDevice(12, pull_up=True)

# Run the motor so encoder should generate signals
kit: MotorKit = MotorKit(address=0x60)
print("Running motor...")
kit.motor1.throttle = 1

# Watch for changes
print("Watching for signal changes for 5 seconds...")
for i in range(50):
    print(f"Green: {green.value}  White: {white.value}", end="\r")
    sleep(0.1)

kit.motor1.throttle = 0.0
print("\n\nDid the values toggle between 0 and 1?")