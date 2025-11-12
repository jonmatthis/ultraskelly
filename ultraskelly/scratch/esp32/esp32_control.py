"""
ESP32 Sinusoidal Motor Driver
=============================

Hardware Setup:
--------------
- ESP32-WROOM board
- DC motor (red/black wires)
- USB cable for programming

Wiring:
-------
Motor red (+)   → GPIO25 (DAC1)
Motor black (-) → GND

Software Setup (One-time):
-------------------------
1. Install tools:
   uv pip install esptool mpremote

2. Download MicroPython firmware:
   https://micropython.org/download/ESP32_GENERIC/
   (Get the latest .bin file, e.g., ESP32_GENERIC-20240105-v1.22.1.bin)

3. Find your device port:
   Windows: Check Device Manager → Ports (COM & LPT)
            Look for "USB-SERIAL CH340" or "CP210x"
            Note: COM3, COM4, etc.

   Linux:   Run: ls /dev/ttyUSB* or ls /dev/ttyACM*
            Usually: /dev/ttyUSB0 or /dev/ttyACM0
            (On RPi5, may need: sudo usermod -a -G dialout $USER, then logout/login)

4. Flash MicroPython to ESP32:

   Windows:
   esptool --port COM3 erase_flash
   esptool --port COM3 --chip esp32 write_flash -z 0x1000 ESP32_GENERIC-20250911-v1.26.1.bin

   Linux:
   esptool.py --port /dev/ttyUSB0 erase_flash
   esptool.py --port /dev/ttyUSB0 --chip esp32 write_flash -z 0x1000 ESP32_GENERIC-20240105-v1.22.1.bin

   (Replace port and filename with your actual values)

Running This Script:
-------------------
Windows:
mpremote connect COM3 run esp32_control.py

Linux:
mpremote connect /dev/ttyUSB0 run esp32_control.py

Or if only one device connected (auto-detect):
mpremote run esp32_control.py

Interactive REPL for testing:
mpremote connect COM3        (Windows)
mpremote connect /dev/ttyUSB0  (Linux)

Press Ctrl+C to stop the motor.
Press Ctrl+D (in REPL) to soft reboot ESP32.

Troubleshooting:
---------------
- "Permission denied" on Linux: sudo chmod 666 /dev/ttyUSB0 or add user to dialout group
- "Port not found": Unplug/replug USB, check drivers
- Motor not smooth: Adjust delay (try delay 5ms) or sine wave speed (change 500.0)
- LED not blinking: Some ESP32 boards have LED on different pin (try GPIO2 or no LED)

Notes:
------
- Script automatically resets DAC on startup - no manual reset needed
- LED pulses in sync with motor to show it's running
- Adjust the divisor in sin() calculation (500.0) to change wave speed
  Smaller = faster, Larger = slower
- Motor draws current - if > 40mA, add a transistor for safety
- DAC output is 0-3.3V (not 5V), but should work for most toy motors
- IDE may show warnings about MicroPython-specific functions - ignore them, code will run fine on ESP32
"""

import math
import time

from machine import Pin, DAC, reset

# Force reset of GPIO25 to clear any previous DAC state
try:
    # Try to reset the pin by initializing as input first
    p = Pin(25, Pin.IN)
    time.sleep_ms(50)
except Exception as e:
    print(f"Pin reset warning: {e}")

# Now initialize DAC - should work clean
try:
    dac = DAC(Pin(25))
except OSError as e:
    if "ESP_ERR_INVALID_STATE" in str(e) or e.args[0] == -259:
        print("DAC in use - performing hard reset...")
        time.sleep(1)
        reset()  # Hard reset the board
        print("Board reset - please rerun the script.")
    raise

# Setup built-in LED (usually GPIO2 on ESP32-WROOM)
led = Pin(2, Pin.OUT)

print("Motor controller starting...")

try:
    while True:
        # Calculate sine wave value
        sine_val = math.sin(time.ticks_ms() / 500.0)
        dac_val = int((sine_val + 1.0) * 127.5)

        # Drive motor with sine wave
        dac.write(dac_val)

        # Pulse LED in sync with motor (LED on when motor > 50% power)
        led.value(1 if dac_val > 127 else 0)

        time.sleep_ms(10)

except KeyboardInterrupt:
    # Clean shutdown on Ctrl+C
    dac.write(0)
    led.value(0)
    print("\nMotor stopped")