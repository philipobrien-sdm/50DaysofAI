# Day 1: Tensor Mechanics \& Gradient Descent from Scratch

\---

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
    def \\\\\\\_\\\\\\\_init\\\\\\\_\\\\\\\_(self, data, requires\\\\\\\_grad=False, \\\\\\\_children=()):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros\\\\\\\_like(self.data, dtype=np.float32) if requires\\\\\\\_grad else None
        self.requires\\\\\\\_grad = requires\\\\\\\_grad
        
        # Dynamic computational graph tracking
        self.\\\\\\\_backward = lambda: None
        self.\\\\\\\_prev = set(\\\\\\\_children)

    def \\\\\\\_\\\\\\\_matmul\\\\\\\_\\\\\\\_(self, other):
        """Matrix Multiplication: self @ other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, \\\\\\\_children=(self, other))

        def \\\\\\\_backward():
            if self.requires\\\\\\\_grad:
                self.grad += out.grad @ other.data.T
            if other.requires\\\\\\\_grad:
                other.grad += self.data.T @ out.grad

        out.\\\\\\\_backward = \\\\\\\_backward
        out.requires\\\\\\\_grad = self.requires\\\\\\\_grad or other.requires\\\\\\\_grad
        return out

    def mean\\\\\\\_squared\\\\\\\_error(self, target):
        """Mean Squared Error Loss: mean((self - target)^2)"""
        diff = self.data - target.data
        loss\\\\\\\_val = np.mean(diff \\\\\\\*\\\\\\\* 2)
        out = Tensor(loss\\\\\\\_val, \\\\\\\_children=(self, target))

        def \\\\\\\_backward():
            if self.requires\\\\\\\_grad:
                N = self.data.size
                self.grad += (2.0 / N) \\\\\\\* diff \\\\\\\* out.grad

        out.\\\\\\\_backward = \\\\\\\_backward
        out.requires\\\\\\\_grad = self.requires\\\\\\\_grad or target.requires\\\\\\\_grad
        return out

    def backward(self):
        """Topologically sorts nodes and runs reverse-mode backpropagation."""
        topo = \\\\\\\[]
        visited = set()

        def build\\\\\\\_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.\\\\\\\_prev:
                    build\\\\\\\_topo(child)
                topo.append(v)

        build\\\\\\\_topo(self)

        # Seed the output node gradient
        self.grad = np.ones\\\\\\\_like(self.data, dtype=np.float32)

        # Backpropagate in topological order
        for node in reversed(topo):
            node.\\\\\\\_backward()

    def zero\\\\\\\_grad(self):
        """Resets accumulated gradients."""
        if self.grad is not None:
            self.grad.fill(0.0)


# --- Execution Example ---

# 1. Dataset setup (Target relationship: y = 2x)
X = Tensor(\\\\\\\[\\\\\\\[1.0], \\\\\\\[2.0], \\\\\\\[3.0]], requires\\\\\\\_grad=False)
y = Tensor(\\\\\\\[\\\\\\\[2.0], \\\\\\\[4.0], \\\\\\\[6.0]], requires\\\\\\\_grad=False)

# 2. Initial Weight
w = Tensor(\\\\\\\[\\\\\\\[0.1]], requires\\\\\\\_grad=True)

# 3. Training Step Loop
lr = 0.05
for epoch in range(10):
    w.zero\\\\\\\_grad()

    # Forward Pass
    pred = X @ w
    loss = pred.mean\\\\\\\_squared\\\\\\\_error(y)

    # Automatic Backward Pass
    loss.backward()

    # Gradient Descent Weight Update
    w.data -= lr \\\\\\\* w.grad

    print(f"Epoch {epoch+1:2d} | Loss: {loss.data:.4f} | Weight: {w.data\\\\\\\[0]\\\\\\\[0]:.4f}")
'''

## 3. Reflection: The Scalability Bottleneck

Why do pure NumPy gradient implementations fall apart at modern AI scales?

> ### 1. Sequential CPU Memory Access
> NumPy operates synchronously on single CPU memory channels (RAM to CPU caches). Large tensor operations quickly run into memory bandwidth limits because CPUs are optimized for low-latency sequential tasks rather than massively parallel matrix math.

> ### 2. The GPU Parallelization Advantage
> Scaled machine learning relies on thousands of operations running simultaneously. Frameworks like PyTorch dispatch computational graphs to CUDA kernels on GPUs, where thousands of lightweight cores compute tensor slices concurrently.

> ### 3. Modern Hardware Optimization \& Kernel Fusion
> Modern frameworks don't just pass individual operations to hardware one by one. They use \*\*kernel fusion\*\* (e.g., PyTorch `torch.compile` or JAX XLA) to combine multiple sequential operations into a single GPU kernel call, eliminating high-bandwidth memory (HBM) write overhead.





