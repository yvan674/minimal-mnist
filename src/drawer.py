"""Drawer.

Allows the user to draw the number instead of using the MNIST dataset.
"""
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from skimage.draw import circle_perimeter_aa


class DrawFrame(tk.Frame):
    def __init__(self, master=None, width=280, height=280):
        super().__init__(master=master, width=width, height=height)
        self.img_array =  np.array([28, 28], int)
        self.img = ImageTk.PhotoImage(image=Image.new('L', (280, 280)))

        # Set up canvas
        self.canvas = tk.Canvas(self, width=280, height=280)
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw",
                                                     image=self.img)
        self.canvas.pack()

    def on_click(self, event):
        """On click action."""
