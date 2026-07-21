# Day 1: Tensor Mechanics \& Gradient Descent from Scratch

## 1\. Theory: The Engine Behind Modern ML

Deep learning models don't just "learn"—they adjust their weights based on how wrong their predictions were. This process relies on four core concepts:

* **Computational Graphs:** Every operation in a neural network forms a directed acyclic graph (DAG). Nodes represent tensors (data/weights) or mathematical operations, while edges represent the flow of data.
* **Partial Derivatives:** When a loss function depends on millions of parameters, partial derivatives ($\\frac{\\partial L}{\\partial w}$) tell us how changing one specific weight $w$ alters the total loss $L$, holding all other weights constant.
* **The Chain Rule:** To calculate gradients deep inside a complex graph, local derivatives are multiplied backwards from the output to the input:
$$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial w}$$
* **Weight Update Mechanics:** Once gradients are calculated, parameters are updated in the opposite direction of the gradient to minimize loss via Gradient Descent:
$$w\_{\\text{new}} = w\_{\\text{old}} - \\eta \\cdot \\frac{\\partial L}{\\partial w}$$
*(where $\\eta$ represents the learning rate).*

\---

## 2\. Hands-on Python: Minimal Micro-Autograd Engine

Below is a functional autograd engine implementing dynamic reverse-mode automatic differentiation in pure NumPy.

```python
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
```

## 3. Reflection: The Scalability Bottleneck

Why do pure NumPy gradient implementations fall apart at modern AI scales?

> ### 1. Sequential CPU Memory Access
> NumPy operates synchronously on single CPU memory channels (RAM to CPU caches). Large tensor operations quickly run into memory bandwidth limits because CPUs are optimized for low-latency sequential tasks rather than massively parallel matrix math.

> ### 2. The GPU Parallelization Advantage
> Scaled machine learning relies on thousands of operations running simultaneously. Frameworks like PyTorch dispatch computational graphs to CUDA kernels on GPUs, where thousands of lightweight cores compute tensor slices concurrently.

> ### 3. Modern Hardware Optimization \& Kernel Fusion
> Modern frameworks don't just pass individual operations to hardware one by one. They use \*\*kernel fusion\*\* (e.g., PyTorch `torch.compile` or JAX XLA) to combine multiple sequential operations into a single GPU kernel call, eliminating high-bandwidth memory (HBM) write overhead.





