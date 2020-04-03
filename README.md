# Minimal MNIST
A very small fully connected neural network able to classify images from the MNIST database.
The purpose of this is to eventually create a physical visualization of the neural network.

The network is optimized using BOHB to find the best hyperparameters. 

# Goals
The goal is to create a very small network that can classify images from MNIST.
The final network will then be implemented on a Raspberry Pi Zero W and the values of every neuron are then to be visualized using physical LEDs

# Best configuration
| Layer | Parameter         | Value             |
|-------|-------------------|-------------------|
| 1     | Input             | Flattened 28 x 28 |
|       | Output dimensions | 16                |
|       | Activation        | ReLU              |
| 2     | Output dimensions | 36                |
|       | Activation        | ReLU              |

# Visualization
Network prediction visualization can be done using `visualizer.py`