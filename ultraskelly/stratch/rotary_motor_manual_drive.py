from time import sleep

from gpiozero import RotaryEncoder

print("Encoder Test - Manually turn the motor shaft")
print("=" * 50)

# Test with your current wiring (green=16, white=12)
encoder: RotaryEncoder = RotaryEncoder(a=16, b=12, max_steps=1000000)
encoder.steps = 0

print("Watching encoder for 10 seconds...")
print("MANUALLY spin the motor shaft and watch the numbers change\n")

for i in range(100):
    pos: int = encoder.steps
    print(f"Position: {pos:6d}", end="\r")
    sleep(0.1)

print(f"\n\nFinal position: {encoder.steps}")

if encoder.steps == 0:
    print("\n⚠ PROBLEM: Encoder never changed!")
    print("\nTroubleshooting:")
    print("1. Are green/white wires actually connected to GPIO 16 and 12?")
    print("2. Try enabling pull-ups (see next test)")
    print("3. Encoder might be damaged or wrong wire colors")
else:
    print("\n✓ Encoder is working!")