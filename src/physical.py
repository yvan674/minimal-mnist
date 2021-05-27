"""Physical.

Code to run the physical visualization of the network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    19 May 2020.
"""
import board
import neopixel
import numpy as np
from math import floor, sin, pi
from time import sleep, time
from inference import AI
from colors import LINEAR_TUPLE
from argparse import ArgumentParser
from startup_sequence import startup
from PIL import Image, ImageTk
import tkinter as tk
import signal
import sys


PIXEL_PIN = board.D12
NUM_PIXELS = 45
BRIGHTNESS = 0.2
ORDER = neopixel.RGB

C_LEN = (0, 10, 15, 10, 10)  # Individual Column Lengths

# Animation Parameters
HALF_PERIOD = 1
OFFSET = (0.8, 1.0, 1.2, 1.4)
PEAK_DURATIONS = (1, 1, 1.7, 1.7)

PIXELS = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS,
                           auto_write=False, pixel_order=ORDER)


def parse_args():
    p = ArgumentParser(description='light up physical LEDs to represent a '
                                   'neural network')
    p.add_argument('ROOT', type=str, help='path to the MNIST dataset')
    p.add_argument('MODEL', type=str, help='path to the NN model')

    return p.parse_args()


def handle_exit(sig, frame):
    # Handle shutdowns gracefully
    raise SystemExit


signal.signal(signal.SIGTERM, handle_exit)


def get_next_values(ai) -> (np.ndarray, int, list, list):
    """Runs inference and parses the output of the network."""
    img, out, h1, h2 = ai.infer_next()
    # Reshape and normalize the layers
    h1 = h1.reshape(C_LEN[1])
    h2 = h2.reshape(C_LEN[2])

    h1 *= 255. / h1.max()
    h2 *= 255. / h2.max()

    h1 = h1.astype(int).tolist()
    h2 = h2.astype(int).tolist()

    return img, out, h1, h2


def animation_running(curr_time: float, half_period=2., offset=0.,
                      peak_duration=0.) -> bool:
    """Calculates based on the given parameters if the animation is finished.

    This function is given the parameters of the final column of LEDs.

    Returns:
        True if animation is not yet finished.
    """
    return curr_time <= (half_period + offset + peak_duration)


def brightness_calc(curr_time: float, half_period=2., offset=0.,
                    peak_duration=0.) -> float:
    """Calculates the color value multiplier at a given time.

    Args:
        curr_time: The current time. Must be greater than 0. 0 is the starting
            time of the current animation.
        half_period: How long the animation lasts in ms. Defaults to 2 s
        offset: How many s to wait after the beginning of the animation before
            starting. Defaults to 0 s.
        peak_duration: How long to hold the peak value in ms. Defaults to 0 s

    Returns:
        A value between 0 and 1.
    """
    if peak_duration == 0.:
        # It's a simple sine function
        return max(0, sin(((pi * curr_time) - (offset * pi)) / half_period))
    else:
        # Then it becomes piecewise function
        x_0 = offset                    # Start of animation
        x_1 = x_0 + (half_period / 2)   # End of rising part
        x_2 = x_1 + peak_duration       # End of peak hold
        x_3 = x_2 + (half_period / 2)   # End of animation

        if x_0 <= curr_time < x_1:
            # The first part (i.e. the rising portion is done like usual
            return brightness_calc(curr_time, half_period, offset, 0)
        elif x_1 <= curr_time < x_2:
            # During the peak duration, always output 1
            return 1.
        elif x_2 <= curr_time < x_3:
            # The descending part of the curve
            return brightness_calc(curr_time, half_period,
                                   offset + peak_duration, 0)
        else:
            # otherwise, stay at 0
            return 0.


def fade_on():
    """Fades to the LED default color state."""
    target_color = LINEAR_TUPLE[0]
    start_time = time()
    duration = HALF_PERIOD / 2
    elapsed_time = 0.
    while elapsed_time <= duration:
        current_color = [floor(color * (elapsed_time / duration))
                         for color in target_color]
        PIXELS.fill(current_color)
        PIXELS.show()
        sleep(0.0005)
        elapsed_time = time() - start_time


def main(root, model):
    # First initialize the LEDs and the screen
    window = tk.Tk()
    window.attributes('-fullscreen', True)
    window.configure(background='black')
    window.config(cursor="none")
    canvas = tk.Canvas(window, width=480, height=320, highlightthickness=0,
                       background='black')
    black_image = Image.new('RGB', (320, 320), 'black')
    canvas_image = canvas.create_image(80, 0, anchor='nw',
                                       image=ImageTk.PhotoImage(black_image))
    canvas.pack()
    window.update()

    fade_on()

    # Then initialize the AI
    ai = AI(root, model)

    # Initial start condition
    playing = False

    while True:
        if not playing:
            # When done with the animation, get next values
            img, out, h0, h1 = get_next_values(ai)

            # Check to see if the image is an image or a numpy array
            if not isinstance(img, Image.Image):
                # If it isn't, then turn it into one and rotate and resize it
                img = Image.fromarray(img)
                img = img.resize((320, 320), Image.NEAREST)
                img = img.rotate(90)
                img = ImageTk.PhotoImage(img)
                canvas.itemconfig(canvas_image, image=img)
                window.update()

            # Make h0, h1, out into a linear array of values
            activations = [0] * 45

            # First assign values to h0 and h1
            activations[:C_LEN[1]] = h0
            activations[C_LEN[1]:C_LEN[1] + C_LEN[2]] = h1

            # Then assign 1 to the correct final output values
            activations[C_LEN[1] + C_LEN[2] + out] = 255
            activations[(out + 1) * -1] = 255

            start_time = time()
            playing = True
        else:
            px_vals = activations.copy()
            curr_time = time() - start_time
            for i in range(4):
                # Iterate by column
                brightness = brightness_calc(curr_time,
                                             HALF_PERIOD,
                                             OFFSET[i],
                                             PEAK_DURATIONS[i])
                px_vals[sum(C_LEN[:i + 1]):sum(C_LEN[:i + 2])] = [
                    floor(i * brightness)
                    for i in px_vals[sum(C_LEN[:i + 1]):sum(C_LEN[:i + 2])]
                ]
                # px_vals is now a value between 0 and 255

            for i in range(len(px_vals)):
                PIXELS[i]=LINEAR_TUPLE[px_vals[i]]

            PIXELS.show()
            playing = animation_running(curr_time, HALF_PERIOD, OFFSET[3],
                                        PEAK_DURATIONS[3])
            sleep(0.0005)


if __name__ == '__main__':
    args = parse_args()
    startup()
    try:
        main(args.ROOT, args.MODEL)
    except (KeyboardInterrupt, SystemExit):
        PIXELS.fill((0, 0, 0))
        PIXELS.show()
        sys.exit()
