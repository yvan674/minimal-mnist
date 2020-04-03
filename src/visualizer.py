"""Visualizer.

Visualizes network results in using tkinter.
"""
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

from inference import AI


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
        self.prediction_label = tk.Label(self, textvariable=self.prediction_var)
        self.prediction_label.grid(row=0, column=0)

    def update_prediction(self, value: str):
        """Updates the value of the prediction."""
        self.prediction_var.set(value)


class NetworkVisualization(tk.Frame):
    def __init__(self, layer_1_neurons, layer_2_neurons, master=None):
        """Class for frames that show the network visualization"""
        super().__init__(master=master, width=430, height=200)

class VisualizerUI:
    def __init__(self):
        """Creates a prediction visualizer GUI."""
        self.root = tk.Tk()
        self.root.title("MNIST FCNetwork Inference")
        self.root.resizable(False, False)

        # Configure the grid geometry
        # -----------------------------------
        # |                   |   PRED VAL  |
        # |       IMAGE       |             |
        # |                   |    next     |
        # -----------------------------------
        self.root.geometry("430x280")
        self.root.columnconfigure(0, minsize=280)
        self.root.columnconfigure(1, minsize=220)
        self.root.rowconfigure(0, minsize=140)
        self.root.rowconfigure(1, minsize=140)

        self.image_frame = ImageFrame(self.root)
        self.image_frame.grid(row=0, column=0, rowspan=2)
        self.pred_frame = PredictionOutput(self.root)
        self.pred_frame.grid(row=0, column=1)

        self.next_button = tk.Button(text="Next", command=self.predict_next)
        self.next_button.grid(row=1, column=1)

        self.ai = AI('E:\Offline Docs\Git\minimal-mnist\MNIST',
                     'E:\Offline Docs\Git\minimal-mnist\\best-model.pth')

        self.root.mainloop()


    def predict_next(self):
        """Predicts the next values."""
        img, pred = self.ai.infer_next()
        self.image_frame.update_image(img)
        self.pred_frame.update_prediction(str(pred))


if __name__ == '__main__':
    v = VisualizerUI()
