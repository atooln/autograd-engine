# autograd-engine

baby autograd engine and neural network framework

This project implements a basic automatic differentiation (autograd) engine and a small neural network library built on top of it.

**`ad.py`**: This file contains the core autograd engine.

- It defines a `Value` class that wraps scalar data.
- `Value` objects keep track of their constituent "children" `Value` objects and the operation that produced them, forming a computation graph (DAG).
- It supports arithmetic operations (`+`, `*`, `-`, `/`, `**`), exponentiation (`exp`), and the hyperbolic tangent (`tanh`) activation function.
- Each `Value` object can store its gradient (`grad`).
- A `backward()` method is implemented to perform reverse-mode automatic differentiation, calculating gradients for all nodes in the computation graph starting from an output node.

**`nn.py`**: This file defines a simple neural network library using the `Value` objects from `ad.py`.

- `Neuron`: A basic neuron unit that computes `tanh(sum(wi*xi) + b)`. Weights (`w`) and bias (`b`) are `Value` objects.
- `Layer`: A collection of `Neuron` objects.
- `MLP` (Multi-Layer Perceptron): A class to create a sequence of `Layer` objects, forming a feed-forward neural network.
- The script includes an example of how to define an MLP, perform a forward pass, calculate a loss, and update the network parameters using gradient descent.
