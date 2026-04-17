import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.x = None

    def forward(self, x):
        """
        x shape: (batch_size, in_features)
        Returns shape: (batch_size, out_features)
        """
        self.x = x
        return self.x @ self.weights + self.bias

    def backward(self, dL_dout, lr):
        """
        dL_dout shape: (batch_size, out_features)
        Returns dL_dx shape: (batch_size, in_features)
        """
        dL_dW = np.einsum("bo,bi->io", dL_dout, self.x)
        dL_db = dL_dout.sum(0, keepdims=True)
        dL_dx = np.einsum("bo,io->bi", dL_dout, self.weights)
        self.weights -= lr * dL_dW
        self.bias -= lr * dL_db

        return dL_dx


def _numerical_grad_x(loss_fn, x, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(x[idx])
        x[idx] = old + h
        fp = loss_fn(x)
        x[idx] = old - h
        fm = loss_fn(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * h)
        it.iternext()
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    batch, in_f, out_f = 3, 5, 4
    x = rng.standard_normal((batch, in_f))
    lin = Linear(in_f, out_f)
    # Deterministic weights/bias for repeatable checks
    lin.weights = rng.standard_normal((in_f, out_f))
    lin.bias = rng.standard_normal((1, out_f))

    out = lin.forward(x)
    assert out.shape == (batch, out_f)
    expected = x @ lin.weights + lin.bias
    assert np.allclose(out, expected, rtol=1e-12, atol=1e-12)

    # dL/dx: L = sum(out^2), dL_dout = 2*out
    dL_dout = 2 * out
    W_before = lin.weights.copy()
    b_before = lin.bias.copy()
    dL_dx = lin.backward(dL_dout, lr=0.0)
    assert np.allclose(lin.weights, W_before) and np.allclose(lin.bias, b_before)
    assert dL_dx.shape == x.shape
    analytic_dx = dL_dout @ lin.weights.T
    assert np.allclose(dL_dx, analytic_dx, rtol=1e-11, atol=1e-11)

    # Numerical check on x for same loss (weights fixed again)
    lin2 = Linear(in_f, out_f)
    lin2.weights = lin.weights.copy()
    lin2.bias = lin.bias.copy()
    x2 = x.copy()

    def loss_x_only(arr):
        layer = Linear(in_f, out_f)
        layer.weights = lin2.weights.copy()
        layer.bias = lin2.bias.copy()
        o = layer.forward(arr)
        return np.sum(o**2)

    num_dx = _numerical_grad_x(loss_x_only, x2.copy(), h=1e-4)
    lin4 = Linear(in_f, out_f)
    lin4.weights = lin2.weights.copy()
    lin4.bias = lin2.bias.copy()
    out4 = lin4.forward(x2)
    analytic_from_mod = lin4.backward(2 * out4, lr=0.0)
    assert np.allclose(analytic_from_mod, num_dx, rtol=5e-4, atol=5e-4)

    print("linear: all tests passed.")
