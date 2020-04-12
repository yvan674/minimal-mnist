"""Visualizer.

Visualizes network results in using tkinter.
"""
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from math import floor

from inference import AI
from colors import LINEAR, DIVERGING
from importlib.util import find_spec

USE_NUMPY = False if find_spec('torch') is not None else True


class ImageFrame(tk.Frame):
    def __init__(self, master=None):
        """Super class for frames that can contain images."""
        super().__init__(master=master, width=280, height=280)

        self.img = ImageTk.PhotoImage(image=Image.new('RGB', (280, 280)))

        # Set up canvas
        self.canvas = tk.Canvas(self, width=280, height=280)
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


class PredictionOutput(tk.Frame):
    def __init__(self, master=None):
        """Super class for frames that show the prediction output."""
        super().__init__(master=master, width=220, height=140)
        self.columnconfigure(0, minsize=220)
        self.rowconfigure(0, minsize=140)

        # Prepare TK variables
        self.prediction_var = tk.StringVar(self, value="-")
        self.pred_label = tk.Label(self, textvariable=self.prediction_var,
                                   font=("Helvetica", "40"))
        self.pred_label.grid(row=0, column=0)

    def update_prediction(self, value: str):
        """Updates the value of the prediction."""
        self.prediction_var.set(value)


class NetworkVisualization(tk.Frame):
    def __init__(self, num_l1, num_l2, fc1_weights, fc2_weights, master=None):
        """Class for frames that show the network visualization"""
        self.height = 800
        self.width = 1000
        super().__init__(master=master, width=self.width, height=self.height)

        self.fc_weights = [fc1_weights, fc2_weights]

        self.current_h = None
        self.current_pred = 0
        self.current_tick = 0

        self.canvas = tk.Canvas(self, width=self.width, height=self.height)

        neurons_per_layer = [num_l1, num_l2, 10]
        x_sep = 60
        diameter = 50
        self.layers = []
        self.connections = []
        self.ca = []  # connection activations

        for i in range(3):
            # Create the regular neurons
            self.layers.append(self.create_neurons(neurons_per_layer[i], i,
                                                   x_sep, diameter))

        for i in range(2):
            # Create the connections between neurons
            self.connections.append(self.create_connections(i))

        # Move the neurons above the connections
        self.canvas.tag_raise('neuron')

        self.canvas.pack()
        self._job = None

    def create_connections(self, layer: int):
        """Creates connections between neurons of the layers."""
        lines = []
        for prev in self.layers[layer]:
            bbox = self.canvas.bbox(prev)
            start_point = (floor((bbox[0] + bbox[2]) / 2),
                           bbox[3])
            for next in self.layers[layer + 1]:
                bbox = self.canvas.bbox(next)
                end_point = (floor((bbox[0] + bbox[2]) / 2),
                             bbox[1])
                coords = (*start_point, *end_point)
                lines.append(self.canvas.create_line(coords,
                                                     fill=DIVERGING[128]))

        return lines

    def create_neurons(self, n: int, row: int, x_sep: int,
                       diameter: int) -> list:
        """Create neurons for each layer."""
        out = []
        # Calculate initial x and initial y to center the circles.
        if n % 2 == 0:
            num_on_each_side = n / 2 - 1
            offset = x_sep * num_on_each_side
            offset += diameter + ((x_sep - diameter) / 2)
        else:
            num_on_each_side = floor(n / 2)
            offset = x_sep * num_on_each_side
            offset += diameter / 2

        initial_x = int(self.width / 2 - offset)
        initial_y = int(((self.height) / 4) * (row + 1))
        initial_y -= int(diameter / 2)

        for i in range(n):
            start_x = initial_x + (x_sep * i)
            start_y = initial_y
            out.append(self.canvas.create_oval(
                ((start_x, start_y), (start_x + diameter, start_y + diameter)),
                fill=LINEAR[0], tag='neuron'
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

        self.current_h = [h1.reshape(len(self.layers[0])),
                          h2.reshape(len(self.layers[1]))]
        self.ca = []
        self.current_pred = pred
        self.current_tick = 0

        for i in range(2):
            # Reshapse to a flat layer, normalizes it, turns it to a value out
            # of 255, then turns it into a numpy array.
            h = self.current_h[i]
            if not USE_NUMPY:
                h = h.numpy()

            # Get connection activations
            fw = self.fc_weights[i]
            activations = h * fw
            activations /= np.max(np.abs(activations)) * 2
            self.ca.append(activations.flatten())
            h /= h.max()

            # Turns it into an integer value, clips it to 0 to 255, and turns it
            # into a python list.
            h = h.clip(0, 1.).tolist()
            self.current_h[i] = h

        for layer in self.layers:
            for neuron in layer:
                self.canvas.itemconfig(neuron, fill=LINEAR[0])

        for layer in self.connections:
            for connection in layer:
                self.canvas.itemconfig(connection, fill=DIVERGING[128])

        self._animate_neurons()

    def _animate_neurons(self):
        """Animates the neurons."""
        step = floor(self.current_tick / 255)
        t_val = self.current_tick % 255
        if step == 4:
            # Final layer animation, aka output layer
            for i in range(10):
                if i == self.current_pred - 1:
                    self.canvas.itemconfig(
                        self.layers[2][i],
                        fill=LINEAR[floor(1. * t_val)])
                else:
                    self.canvas.itemconfig(self.layers[2][i], fill=LINEAR[0])
        elif step in (0, 2):
            # Normal neurons
            layer = int(step / 2)
            for j, n in enumerate(self.current_h[layer]):
                self.canvas.itemconfig(self.layers[layer][j],
                                       fill=LINEAR[floor(n * t_val)])

        elif step in (1, 3):
            # Animations for connections
            layer = int((step - 1) / 2)
            for j, val in enumerate(self.ca[layer]):
                self.canvas.itemconfig(self.connections[layer][j],
                                       fill=DIVERGING[floor(val * t_val) + 128])


        # Add to tick if still animating
        if step < 5:
            self.current_tick += 12
            self._job = self.after(1, self._animate_neurons)
        else:
            # Done animating the current set
            self.current_tick = 0


class VisualizerUI:
    def __init__(self):
        """Creates a prediction visualizer GUI."""
        self.root = tk.Tk()
        self.root.title("MNIST FCNetwork Inference")
        self.root.resizable(False, False)
        self.bf = tk.Frame(self.root)  # Stands for background frame

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
        self.bf.columnconfigure(0, minsize=280)
        self.bf.columnconfigure(1, minsize=220)
        self.bf.rowconfigure(0, minsize=140)
        self.bf.rowconfigure(1, minsize=140)

        self.image_frame = ImageFrame(self.bf)
        self.image_frame.grid(row=0, column=0, rowspan=2)
        self.pred_frame = PredictionOutput(self.bf)
        self.pred_frame.grid(row=0, column=1)

        # This is the button frame
        self.buttons = tk.Frame(self.bf)
        self.next = tk.Button(self.buttons, text="Next", command=self.get_next)
        self.next.grid(row=0, column=0)
        self.play_var = tk.StringVar(self.buttons, value='Play')
        self.play = tk.Button(self.buttons, textvariable=self.play_var,
                              command=self.play)
        self.play.grid(row=1, column=0)
        self.buttons.grid(row=1, column=1)

        if USE_NUMPY:
            model_weights = 'E:\\Offline Docs\\Git\\minimal-mnist\\best-mo' \
                            'del.npy'
        else:
            model_weights = 'E:\\Offline Docs\\Git\\minimal-mnist\\7.pth'
        self.ai = AI('E:\\Offline Docs\\Git\\minimal-mnist\\MNIST',
                     model_weights)

        self.network_vis = NetworkVisualization(self.ai.layer_1_neurons,
                                                self.ai.layer_2_neurons,
                                                self.ai.fc1_weight,
                                                self.ai.fc2_weight,
                                                self.bf)
        self.network_vis.grid(row=2, column=0, columnspan=2)
        self.bf.pack()
        self.playing = False
        self._job = []

        self.root.mainloop()

    def play(self):
        """Plays endlessly until play is pressed again"""
        self.playing = not self.playing
        if not self.playing:
            self.play_var.set('Play')
            if self._job is not None:
                self.root.after_cancel(self._job)
                self._job = None
        else:
            self.play_var.set('Stop')
            self._job = self.root.after(1, self.get_next)

    def get_next(self):
        """Predicts the next values."""
        img, pred, h1, h2 = self.ai.infer_next()
        self.image_frame.update_image(img)
        self.pred_frame.update_prediction(str(pred))
        self.network_vis.update_neurons(h1, h2, pred)

        if self.playing:
            self._job = self.root.after(2000, self.get_next)


if __name__ == '__main__':
    v = VisualizerUI()
