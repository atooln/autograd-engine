"""
NEURAL NETWORK USING CUSTOM AUTOGRAD CLASS
"""
from ad import Value
import random


class Neuron:
    """
    Constructs our base Neuron object using our Value object we created in ad.py
    """
    def __init__(self, nindim:int):
        """
        nindim is the input dimension that the neuron expects
            -so when we do the calculation w*x+b we can resolve every weight with a respective
                input (see __call__)

        - the weights that correspond with the appropriate input dim
        - bias is a constant term which is added after we sum w*x (improves performance)
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(nindim)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x:list[float])-> Value:
        """
        calculates w*x+b and normalizes using an activation function
        """
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        out = act.tanh()
        return out

    def parameters(self)->list[Value]:
        """
        return all of the weights and biases so that we can update grads efficiently
        """
        return self.w + [self.b]

class Layer:
    """
    Convenient structure for creating a list of neurons and evaluating them with the input
    """
    def __init__(self, nindim:int, noutdim:int):
        """
        init the list of neurons  with the input dim
        """
        self.neurons = [Neuron(nindim) for _ in range(noutdim)]

    def __call__(self, x: list[float])-> list[Value]:
        """
        evaluate each neuron in the layer
        """
        l = [n(x) for n in self.neurons]
        return l[0] if len(l) == 1 else l

    def parameters(self)->list[Value]:
        """
        get all of the weights and biases for all neurons and return it as a 1D array
        """
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """
    Creates an MLP (essentially a list of layer objects) and evaluates every layer sequentially
    """
    def __init__(self, nindim:int, nouts:list[int]):
        """
        we create an array called dims which takes the inputdim which is an int and add it to the
        output array
            - note that output array is n dimensional (given that we have n layers)

        when we have a central array like dims we can "connect" each layer together by using the
        output of the previous layer as the input of the current layer
            - this is what self.layers is


        """
        dims = [nindim] + nouts
        self.layers = [Layer(dims[i], dims[i+1]) for i in range(len(nouts))]

    def __call__(self, x:list[float])->Value:
        """
        for each layer, we evaluate it with the input
        """
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self)->list[Value]:
        """
        now get all of the parameters for every layer and keep it in a 1D array
        """
        return [p for l in self.layers for p in l.parameters()]





if __name__ == "__main__":
    print("BASIC MLP TRAINING W/ GRADIENT DESCENT")
    print("Layers (3): 4, 4, 1")

    n = MLP(3, [4,4,1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]


    alpha = 0.001
    print(f"Learning Rate: {alpha}")
    for e in range(100):
        # FORWARD PASS
        ypred = [n(x) for x in xs]
        loss = sum([(yp-y)**2 for yp, y in zip(ypred,ys)])
        print(f"\t(Epoch {e}) loss: {loss.data}")

        # IMPORTANT: flush gradients before each update
        for p in n.parameters():
            p.grad = 0.0
        # backpropagation
        loss.backward()

        # update parameters
        for p in n.parameters():
            p.data += -alpha * p.grad
