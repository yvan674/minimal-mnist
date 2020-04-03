"""Visualizer.

Visualizes network results in using tkinter.
"""
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from math import floor

from inference import AI


class WhiteFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        """Super class for all frames that use a white background."""
        super().__init__(master=master, **kwargs)
        super().configure(background='#ffffff')


class WhiteCanvas(tk.Canvas):
    def __init__(self, master=None, **kwargs):
        """Super class for all white canvases."""
        super().__init__(master=master, **kwargs)
        super().configure(background='#ffffff')


class WhiteLabel(tk.Label):
    def __init__(self, master=None, **kwargs):
        """Super class for all white labels."""
        super().__init__(master=master, **kwargs)
        super().configure(background='#ffffff')


class ImageFrame(WhiteFrame):
    def __init__(self, master=None):
        """Super class for frames that can contain images."""
        super().__init__(master=master, width=280, height=280)

        self.img = ImageTk.PhotoImage(image=Image.new('RGB', (280, 280)))

        # Set up canvas
        self.canvas = WhiteCanvas(self, width=280, height=280)
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw",
                                                     image=self.img)
        self.canvas.pack()

    def update_image(self, image: np.ndarray):
        """Updates the image with the given image array.

        Args:
            image: A numpy array with size [28, 28]
        """
        img = Image.fromarray(image).resize((280, 280), Image.NEAREST)
        self.img = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.canvas_image, image=self.img)


class PredictionOutput(WhiteFrame):
    def __init__(self, master=None):
        """Super class for frames that show the prediction output."""
        super().__init__(master=master, width=220, height=140)
        self.columnconfigure(0, minsize=220)
        self.rowconfigure(0, minsize=140)

        # Prepare TK variables
        self.prediction_var = tk.StringVar(self, value="-")
        self.pred_label = WhiteLabel(self, textvariable=self.prediction_var,
                                     font=("Helvetica", "40"))
        self.pred_label.grid(row=0, column=0)

    def update_prediction(self, value: str):
        """Updates the value of the prediction."""
        self.prediction_var.set(value)


class NetworkVisualization(WhiteFrame):
    def __init__(self, num_l1, num_l2, master=None):
        """Class for frames that show the network visualization"""
        self.height = 500
        self.width = 1000
        super().__init__(master=master, width=self.width, height=self.height)

        self.current_h = None
        self.current_pred = 0
        self.current_tick = 0

        self.canvas = WhiteCanvas(self, width=self.width, height=self.height)

        neurons_per_layer = [num_l1, num_l2, 10]
        x_sep = 24
        diameter = 18
        y_pad = 50
        self.layers = []

        for i in range(3):
            self.layers.append(self.create_neurons(neurons_per_layer[i], i,
                                                   x_sep, diameter, y_pad))

        self.canvas.pack()
        self._job = None

    def create_neurons(self, n: int, row: int, x_sep: int,
                       diameter: int, y_pad: int) -> list:
        out = []
        # Calculate initial x and initial y to center the circles.
        if n % 2 == 0:
            num_on_each_side = n / 2 - 1
            offset = (x_sep) * num_on_each_side
            offset += diameter + ((x_sep - diameter) / 2)
        else:
            num_on_each_side = floor(n / 2)
            offset = (x_sep) * num_on_each_side
            offset += diameter / 2

        initial_x = int(self.width / 2 - offset)
        initial_y = int((self.height - 2 * y_pad) / 2 * (row))
        initial_y -= int(diameter / 2)
        initial_y += y_pad

        for i in range(n):
            start_x = initial_x + (x_sep * i)
            start_y = initial_y
            out.append(self.canvas.create_oval(
                ((start_x, start_y), (start_x + diameter, start_y + diameter)),
                fill='#000000'
            ))
        return out

    @staticmethod
    def l_to_hex(l: int) -> str:
        """Translates a luminosity value in range[0, 255] to hex."""
        rgb = (l, l, l)
        return "#%02x%02x%02x" % rgb

    def update_neurons(self, h1, h2, pred):
        """Update values of the neurons.

        Args:
            h1 (torch.Tensor): Tensor representing outputs of layer 1.
            h2 (torch.Tensor): Tensor representing outputs of layer 2.
            pred (int): Int representing what the output answer is.
        """
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

        self.current_h = [h1, h2]
        for i in range(len(self.current_h)):
            # Reshapse to a flat layer, normalizes it, turns it to a value out of
            # 255, then turns it into a numpy array.
            h = self.current_h[i]
            h = (h.reshape(len(self.layers[i])) / h.max()).numpy()

            # Turns it into an integer value, clips it to 0 to 255, and turns it
            # into a python list.

            h = h.clip(0, 1.).tolist()
            self.current_h[i] = h

        for layer in self.layers:
            for neuron in layer:
                self.canvas.itemconfig(neuron, fill='#000000')

        self.current_pred = pred
        self.current_tick = 0
        self._animate_neurons()

    def _animate_neurons(self):
        """Animates the neurons."""
        layer = floor(self.current_tick / 255)
        t_val = self.current_tick % 255
        if layer < 2:
            for j, n in enumerate(self.current_h[layer]):
                self.canvas.itemconfig(self.layers[layer][j],
                                       fill=self.l_to_hex(floor(n * t_val)))
        elif layer == 2:
            for i in range(10):
                if i == self.current_pred - 1:
                    self.canvas.itemconfig(
                        self.layers[2][i],
                        fill=self.l_to_hex(floor(1 * t_val)))
                else:
                    self.canvas.itemconfig(self.layers[2][i], fill='#000000')
        if layer < 3:
            self.current_tick += 3
            self._job = self.after(1, self._animate_neurons)
        else:
            self.current_tick = 0


class VisualizerUI:
    def __init__(self):
        """Creates a prediction visualizer GUI."""
        self.root = tk.Tk()
        self.root.configure(background='#ffffff')
        self.root.title("MNIST FCNetwork Inference")
        self.root.resizable(False, False)
        self.bf = WhiteFrame(self.root)  # Stands for background frame

        # Configure the grid geometry
        # -----------------------------------
        # |                   |   PRED VAL  |
        # |       IMAGE       |             |
        # |                   |    next     |
        # -----------------------------------
        # |      Network Visualization      |
        # |                                 |
        # -----------------------------------
        # self.root.geometry("500x430")
        self.bf.columnconfigure(0, minsize=500)
        self.bf.columnconfigure(1, minsize=500)
        self.bf.rowconfigure(0, minsize=140)
        self.bf.rowconfigure(1, minsize=140)
        self.bf.rowconfigure(2, minsize=300)

        self.image_frame = ImageFrame(self.bf)
        self.image_frame.grid(row=0, column=0, rowspan=2)
        self.pred_frame = PredictionOutput(self.bf)
        self.pred_frame.grid(row=0, column=1)

        self.next_button = tk.Button(self.bf, text="Next",
                                     command=self.predict_next)
        self.next_button.grid(row=1, column=1)

        self.network_vis = NetworkVisualization(16, 36, self.bf)
        self.network_vis.grid(row=2, column=0, columnspan=2)
        self.bf.pack()


        self.ai = AI('E:\Offline Docs\Git\minimal-mnist\MNIST',
                     'E:\Offline Docs\Git\minimal-mnist\\best-model.pth')

        self.root.mainloop()


    def predict_next(self):
        """Predicts the next values."""
        img, pred, h1, h2 = self.ai.infer_next()
        self.image_frame.update_image(img)
        self.pred_frame.update_prediction(str(pred))
        self.network_vis.update_neurons(h1, h2, pred)


if __name__ == '__main__':
    v = VisualizerUI()
