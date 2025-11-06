from adafruit_motorkit import MotorKit
from gpiozero import RotaryEncoder
from time import sleep

# Initialize motor HAT
kit: MotorKit = MotorKit()

# Initialize encoder (green=GPIO16, white=GPIO12)
encoder: RotaryEncoder = RotaryEncoder(a=16, b=12, max_steps=1000000)

# Zero the position
encoder.steps = 0

print("Running motor forward and watching encoder...")
kit.motor1.throttle = 0.3

for i in range(100):
    print(f"Encoder position: {encoder.steps}")
    sleep(0.1)

kit.motor1.throttle = 0.0
print(f"Final position: {encoder.steps}")