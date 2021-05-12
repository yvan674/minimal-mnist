"""Startup Sequence

Runs a wave sequence of lights from left to right letting us know that the
system is ready.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    May 12, 2021
"""
import board
import neopixel
from math import sin, pi, floor
from time import sleep, time

SLEEP_DURATION = 1
PIXEL_PIN = board.D12
NUM_PIXELS = 45
BRIGHTNESS = 0.2
ORDER = neopixel.RGB

LED_COLUMNS = (0, 10, 25, 35, 45)

PIXELS = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS,
                           auto_write=False, pixel_order=ORDER)


def startup():
    sleep(3)
    start_time = time()
    time_delta = 0.
    # 1.477 is where sin(pi * x - 1.5) intercepts the y-axis
    while time_delta < 1.477:
        time_delta = time() - start_time
        for column in range(4):
            # Calculate brightness of a column for the given frame
            frame_brightness = floor(255. * sin((pi * time_delta)
                                                - (float(column) * 0.5)))
            # Make sure it's positive
            frame_brightness = max(frame_brightness, 0)
            for pixel_num in range(LED_COLUMNS[column],
                                   LED_COLUMNS[column + 1]):
                # set all pixels of the column one by one
                PIXELS[pixel_num] = (frame_brightness, frame_brightness,
                                     frame_brightness)
        PIXELS.show()
        sleep(0.0005)

    # Turn off all LEDs just in case
    for i in range(LED_COLUMNS[-1]):
        PIXELS[i] = (0, 0, 0)


if __name__ == '__main__':
    startup()
