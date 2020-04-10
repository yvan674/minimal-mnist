# Minimal MNIST
A very small fully connected neural network able to classify images from the MNIST database.
The purpose of this is to eventually create a physical visualization of the neural network.

The network is optimized using BOHB to find the best hyperparameters. 

# Goals
The goal is to create a very small network that can classify images from MNIST.
The final network will then be implemented on a Raspberry Pi Zero W and the values of every neuron are then to be visualized using physical LEDs

# Chosen configuration
The following configuration yields a validation accuracy of 94.5% after just 10 epochs.

| Item                      | Parameter         | Value              |
|---------------------------|-------------------|--------------------|
| 1st Fully Connected Layer | Input             | Flattened 28 x 28  |
|                           | Output dimensions | 11                 |
|                           | Activation        | ReLU               |
| 2nd Fully Connected Layer | Output dimensions | 11                 |
|                           | Activation        | ReLU               |
| Optimizer                 | Type              | SGD                |
|                           | Learning Rate     | 0.0038795787201773 |
|                           | Momentum          | 0.9409782496856666 |

# Visualization
Model prediction visualization can be done using `visualizer.py`.
This script visualizes the input from MNIST, the network prediction, and the network activations in the model.
