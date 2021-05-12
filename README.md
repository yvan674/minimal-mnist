# Minimal MNIST

A very small fully connected neural network able to classify images from the MNIST database.
The purpose is to better understand how to use the `hpoptim` package and experimenting with creating an architecture using bayesian optimization techniques.

The network is optimized using Bayesian Optimization and HyperBand (BOHB) to find the best hyperparameters.

## Goals

The goal is to create a very small network that can classify images from MNIST.
The number of output dimensions and activations for the first two layers, as well as the optimizer type, learning rate, and momentum are chosen using BOHB.

## Best configuration

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

## Physical Model

Owing to the simplicity of the model, I have chosen to create a physical version of this neural network.
The hardware for this can be found in the [physical-nn](github.com/yvan674/physical-nn) repo.

For the physical neural network representation, I chose to use a 10-15-10 layer architecture (i.e. output dim of 10, 15, 10 on the first, second and final layers respectively) as it looked more aesthetically appealing.

### Numpy and Torch

As the plan for running the model physically will use a Raspberry Pi Zero W, a numpy version of the model has also been included.

## Visualization

Model prediction visualization can be done using `visualizer.py`.
This script visualizes the input from MNIST, the network prediction, and the network activations in the model.

## Dependencies

- numpy>=1.18
- pillow~=7.1
- pytorch>=1.5
