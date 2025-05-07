"""
MICROGRAD
===================

Objectives:
    1. Create an Object (Value) which allows us to store data
        a. We also want to be able to do (+) and (*) operations on Value Objects
    2. Keep track of the prior Value objects used to create the current Value object (store children)
        a. Record the operations used to obtain the current
    * remark: step 2 basically creates a DAG for obtaining a value
    3. Keep track of the gradient
    4. Implement the backwards pass behavior for multiply, add, and any other function (activation functions like tanh, etc)
    5. Create the recursive backwards pass function (basically topsort on the constructed graph)
    6. The rest is optional, but are nice QOL improvements

"""
import math

class Value:
    def __init__(self, data:float, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._op = op
        self._prev = set(children)
        self._backward = lambda: None

    def __add__(self, other)->object:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other)->object:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self*other**-1

    def __pow__(self, other):
        assert isinstance(other, (int,float)), "other must be a int or float"
        out = Value(self.data**other, (self,), f'**{other}')

        def backward():
            self.grad += other * (self.data **(other-1)) * out.grad

        out._backward = backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def backward():
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    def tanh(self)->object:
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def backward():
            self.grad += (1-t**2) * out.grad
        out._backward = backward
        return out

    def backward(self):
        top, visited = [], set()

        def build(n)-> None:
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    build(child)
                top.append(n)

        build(self)
        self.grad = 1.0

        for n in top[::-1]:
            n._backward()

    def __repr__(self) -> str:
        return f"Value({self.data})"




if __name__ == "__main__":
    # basic perceptron
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.88137)

    x1w1, x2w2 = x1*w1, x2*w2
    x1w1x2w2 = x1w1 + x2w2

    n = x1w1x2w2 + b
    o = n.tanh()
    o.backward()

    print(f"{o}, grad: {o.grad}")
