import numpy as np

class Tensor:
    """
    A minimal autograd Tensor implementation wrapped around NumPy arrays.
    """
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self.requires_grad = requires_grad
        
        # Internal variables for building the dynamic computational graph
        self._backward = lambda: None
        self._prev = set(_children)

    def __matmul__(self, other):
        """Matrix Multiplication: self @ other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _backward():
            # Gradients with respect to matrix multiplication:
            # dL/dA = dL/dOut @ B^T
            # dL/dB = A^T @ dL/dOut
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __sub__(self, other):
        """Subtraction: self - other"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def mean_squared_error(self, target):
        """Mean Squared Error Loss: mean((self - target)^2)"""
        diff = self.data - target.data
        loss_val = np.mean(diff ** 2)
        out = Tensor(loss_val, _children=(self, target))

        def _backward():
            if self.requires_grad:
                # Derivative of MSE wrt prediction: (2 / N) * (pred - target)
                N = self.data.size
                self.grad += (2.0 / N) * diff * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad or target.requires_grad
        return out

    def backward(self):
        """Topologically sorts nodes and runs backpropagation."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Root node seed gradient
        self.grad = np.ones_like(self.data, dtype=np.float32)

        # Backpropagate through the graph in reverse order
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Resets gradients to avoid accumulation across training steps."""
        if self.grad is not None:
            self.grad.fill(0.0)


# --- Training Loop Execution ---

# 1. Dataset setup (y = 2x)
X = Tensor([[1.0], [2.0], [3.0]], requires_grad=False)
y = Tensor([[2.0], [4.0], [6.0]], requires_grad=False)

# 2. Model Weight (Initialize with 0.1)
w = Tensor([[0.1]], requires_grad=True)

lr = 0.05

print("Starting Auto-Diff Training Loop...\n")
for epoch in range(10):
    # Zero gradients before forward pass
    w.zero_grad()

    # Forward Pass
    pred = X @ w
    loss = pred.mean_squared_error(y)

    # Automatic Backward Pass
    loss.backward()

    # Gradient Descent Step
    w.data -= lr * w.grad

    print(f"Epoch {epoch+1:2d} | Loss: {loss.data:.4f} | Weight: {w.data[0][0]:.4f} | Grad: {w.grad[0][0]:.4f}")