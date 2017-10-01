# Multiplication neuron
from typing import Callable, List, NamedTuple
import numpy as np


ScalarField = Callable[[List[float]], float]

# Derivative functions
def derivative_name(f: Callable[..., float],
                    respect_to: str,
                    h: float = 0.0001,
                    **kwargs) -> float:
    """
    ∂f(x,y)   f(x+h,y)-f(x,y)
    ------- = ---------------
      ∂x             h
    """
    if respect_to not in kwargs:
        raise ValueError('Not derivating a variable')
    else:
        changed = kwargs.copy()
        changed[respect_to] += h
        return (f(**changed) - f(**kwargs)) / h

def derivative(fun: ScalarField,
               respect_to: int,
               *args: float,
               h: float = 1e-5) -> float:
    if respect_to >= len(args):
        raise ValueError(f'Function has only {len(args)} variables')
    else:
        changed = list(args)
        changed[respect_to] += h
        return (fun(*changed) - fun(*args)) / h

def numeric_gradient(fun: ScalarField, *args: float) -> List[float]:
    """
         ∂f   ∂f   ∂f
    ∇f = -- + -- + --
         ∂x   ∂y   ∂z
    """
    return [derivative(fun, pos, *args)
            for pos in range(len(args))]

class Gate2:
    def __init__(self, fun, in_cables, out_cable):
        self.fun = fun
        self.in_cables = in_cables
        self.out_cable = out_cable

    def forward(self, *args):
        self.output.val = fun(*args)

    def backward(self):
        for con in self.inputs:
            con.dev += derivative(self, fun, respect_to=con, *self.args)

def forward_pass(layer):
    for cable in layer:
        gate = cable.out_gate
        gate.out_cable = out = gate.f((g.val for g in gate.inputs))

# Fully connected neural network
# Input layers has len(input) size
# Output size has len(classes)
def create_model(X, y, cycles=1000):
    in_layer = X
    syn0 = 2 * np.random.random([X.shape[1]]) - 1
    for _ in range(cycles):
        l1 = sigmoid(np.dot(in_layer, syn0))
        l1_error = y - l1
        l1_delta = l1_error * sigmoid_der(l1)
        print(syn0, l1_delta)
        syn0 += np.dot(X, l1_delta)
    return l1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    print(create_model(X, y))
