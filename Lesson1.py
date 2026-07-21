import numpy as np


class Tensor:
    """
    A minimal autograd Tensor implementation wrapped around NumPy arrays.
    """

    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.grad = (
            np.zeros_like(self.data, dtype=np.float32)
            if requires_grad
            else None
        )

        self._prev = set(_children)
        self._backward = lambda: None

    def __matmul__(self, other):
        """Matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)

        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(
            self.data @ other.data,
            requires_grad=requires_grad,
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def mean_squared_error(self, target):
        """Mean Squared Error Loss."""
        target = target if isinstance(target, Tensor) else Tensor(target)

        diff = self.data - target.data
        loss_val = np.mean(diff ** 2)

        requires_grad = self.requires_grad or target.requires_grad

        out = Tensor(
            loss_val,
            requires_grad=requires_grad,
            _children=(self, target),
        )

        def _backward():
            if self.requires_grad:
                N = self.data.size
                self.grad += (2.0 / N) * diff * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Reverse-mode automatic differentiation."""

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        if self.grad is None:
            self.grad = np.ones_like(self.data, dtype=np.float32)
        else:
            self.grad.fill(1.0)

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.0)


# ----------------------------------------------------
# Example: Learn y = 2x
# ----------------------------------------------------

X = Tensor([[1.0], [2.0], [3.0]])
y = Tensor([[2.0], [4.0], [6.0]])

w = Tensor([[0.1]], requires_grad=True)

lr = 0.05

for epoch in range(20):
    w.zero_grad()

    pred = X @ w
    loss = pred.mean_squared_error(y)

    loss.backward()

    w.data -= lr * w.grad

    print(
        f"Epoch {epoch+1:2d} | "
        f"Loss: {loss.data:.6f} | "
        f"Weight: {w.data[0,0]:.6f}"
    )
