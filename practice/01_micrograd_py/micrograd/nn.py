from micrograd.engine import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))
    def __call__(self, x):
        activation = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.b
        return activation.relu()
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
    def __init__(self, n_input, n_outputs):
        sz = [n_input] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outputs))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]