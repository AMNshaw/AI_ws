from microtorch_py.engine import Tensor
import random
import numpy as np

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.1, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def parameters(self):
        return [self.weight, self.bias]

    def __call__(self, x):
        return x @ self.weight + self.bias

class MLP(Module):
    def __init__(self, n_input, n_output, hiddens):
        self._n_input = n_input
        self._n_output = n_output
        self._hiddens = hiddens
        sz = [n_input] + hiddens + [n_output]
        self.layers = [Linear(sz[i], sz[i + 1]) for i in range(len(sz) - 1)]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x