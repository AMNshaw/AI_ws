import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True, parent=(), op=''):
        self._data = np.asarray(data, dtype=float)
        self._requires_grad = requires_grad
        self._grad = np.zeros_like(self._data)
        self._prev = set(parent)
        self._backward = lambda : None
        self._op = op

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, value):
        self._data = np.asarray(value, dtype=float)

    @property
    def grad(self):
        return self._grad
    @grad.setter
    def grad(self, value):
        self._grad = np.asarray(value, dtype=float)

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def prev(self):
        return self._prev
    @prev.setter
    def prev(self, value):
        self._prev = set(value)

    def zero_grad(self):
        self._grad = np.zeros_like(self._data)

    def _as_same_shape(self, grad, shape):
        grad = np.asarray(grad, dtype=float)
        if grad.shape == shape:
            return grad
        if grad.size == 1:
            return np.full(shape, float(grad.reshape(())))
        if shape == ():
            return grad.sum()
        return grad

    def backward(self):
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor.prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for tensor in reversed(topo):
            tensor._backward()

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), 'tanh')

        def backward_func():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad

        out._backward = backward_func
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), self.requires_grad, (self,), 'ReLU')

        def backward_func():
            if self.requires_grad:
                self.grad += (out.data > 0) * out.grad

        out._backward = backward_func
        return out

    def sigmoid(self):
        z = np.clip(self.data, -500, 500)
        out = Tensor(1.0 / (1.0 + np.exp(-z)), self.requires_grad, (self,), 'sigmoid')

        def backward_func():
            if self.requires_grad:
                self.grad += out.data * (1 - out.data) * out.grad

        out._backward = backward_func
        return out

    def log(self):
        clipped = np.clip(self.data, 1e-8, 1.0 - 1e-8)
        out = Tensor(np.log(clipped), self.requires_grad, (self,), 'log')

        def backward_func():
            if self.requires_grad:
                self.grad += out.grad / clipped

        out._backward = backward_func
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), '+')

        def backward_func():
            if self.requires_grad:
                self.grad += self._as_same_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += other._as_same_shape(out.grad, other.data.shape)

        out._backward = backward_func
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), '*')

        def backward_func():
            if self.requires_grad:
                self.grad += self._as_same_shape(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += other._as_same_shape(out.grad * self.data, other.data.shape)

        out._backward = backward_func
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), '@')

        def backward_func():
            if self.requires_grad:
                if self.data.ndim == 1 and other.data.ndim == 2:
                    self.grad += out.grad @ other.data.T
                elif self.data.ndim == 2 and other.data.ndim == 1:
                    self.grad += np.outer(out.grad, other.data)
                else:
                    self.grad += out.grad @ other.data.T
            if other.requires_grad:
                if self.data.ndim == 1 and other.data.ndim == 2:
                    other.grad += np.outer(self.data, out.grad)
                elif self.data.ndim == 2 and other.data.ndim == 1:
                    other.grad += self.data.T @ out.grad
                else:
                    other.grad += self.data.T @ out.grad

        out._backward = backward_func
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), self.requires_grad, (self,), 'sum')

        def backward_func():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = backward_func
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, self.requires_grad, (self,), f'**{other}')

        def backward_func():
            if self.requires_grad:
                self.grad += other * self.data**(other - 1) * out.grad

        out._backward = backward_func
        return out

    def __neg__(self):
        return self * (-1)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"