

import time

from adafruit_servokit import ServoKit

time.sleep(5)
kit= ServoKit(channels=16)
kit.servo[3].angle =180
kit.servo[7].angle =180
kit.servo[11].angle =180
time.sleep(1)
kit.servo[3].angle =0
kit.servo[7].angle =0
kit.servo[11].angle =0
time.sleep(1)
kit.servo[3].angle =90
kit.servo[7].angle =90
kit.servo[11].angle =90
