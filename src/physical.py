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
from math import floor
from time import sleep
from inference import AI
from colors import LINEAR_TUPLE
from argparse import ArgumentParser
from startup_sequence import startup


SLEEP_DURATION = 1
PIXEL_PIN = board.D12
NUM_PIXELS = 45
BRIGHTNESS = 0.2
ORDER = neopixel.RGB

NN_LAYERS = [10, 15, 10]

PIXELS = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS,
                           auto_write=False, pixel_order=ORDER)


def parse_args():
    p = ArgumentParser(description='light up physical LEDs to represent a '
                                   'neural network')
    p.add_argument('ROOT', type=str, help='path to the MNIST dataset')
    p.add_argument('MODEL', type=str, help='path to the NN model')

    return p.parse_args()


def get_next_values(ai) -> (np.ndarray, int, list, list):
    """Runs inference and parses the output of the network."""
    img, out, h1, h2 = ai.infer_next()
    # Reshape and normalize the layers
    h1 = h1.reshape(NN_LAYERS[0])
    h2 = h2.reshape(NN_LAYERS[1])

    h1 *= 255. / h1.max()
    h2 *= 255. / h2.max()

    h1 = h1.astype(int).tolist()
    h2 = h2.astype(int).tolist()

    return img, out, h1, h2


def main(root, model):
    ai = AI(root, model)

    playing = True
    current_tick = 0
    img, out, h0, h1 = get_next_values(ai)

    while True:
        if not playing:
            # If it's done playing the animation, do the next one after 1 second
            sleep(SLEEP_DURATION)
            img, out, h0, h1 = get_next_values(ai)
            playing = True
        else:
            step = int(floor(current_tick / 255))
            t_val = current_tick % 255

            if step == 2:
                # Finish layer animation, aka output layer
                for i in range(NN_LAYERS[2]):
                    px_val = NN_LAYERS[0] + NN_LAYERS[1] + i
                    if i == out:
                        color = LINEAR_TUPLE[floor(t_val)]
                        PIXELS[px_val] = color       # Neuron
                        PIXELS[px_val + 10] = color  # Number
                    else:
                        PIXELS[px_val] = LINEAR_TUPLE[0]       # Neuron
                        PIXELS[px_val + 10] = LINEAR_TUPLE[0]  # Number
            elif step < 2:
                for i in range(NN_LAYERS[step]):
                    if step == 0:
                        px_val = i
                        h = h0
                    else:
                        px_val = NN_LAYERS[0] + i
                        h = h1

                    PIXELS[px_val] = LINEAR_TUPLE[floor(
                        (h[i] / 255.) * t_val
                    )]

            # Add to tick if still animating and actually color pixels
            if step < 3:
                PIXELS.show()
                current_tick += 6
                sleep(0.00005)
            else:
                current_tick = 0
                playing = False


if __name__ == '__main__':
    args = parse_args()
    startup()
    sleep(2)
    main(args.ROOT, args.MODEL)
