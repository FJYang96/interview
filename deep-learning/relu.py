import numpy as np


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dL_dout):
        return dL_dout * (self.x > 0)


def _numerical_grad_relu_x(f_loss, x, h=1e-5):
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(x[idx])
        x[idx] = old + h
        fp = f_loss(x)
        x[idx] = old - h
        fm = f_loss(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * h)
        it.iternext()
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    relu = ReLU()
    x = rng.standard_normal((2, 5))
    y = relu.forward(x)
    assert np.allclose(y, np.maximum(0, x))
    dout = rng.standard_normal(x.shape)
    dx = relu.backward(dout)
    assert np.allclose(dx, dout * (x > 0))

    # L = sum(ReLU(x)^2); dL/dx = 2*ReLU(x) * (x>0)
    x2 = rng.standard_normal((2, 3))

    def loss_relu_sq(arr):
        r = ReLU()
        out = r.forward(arr)
        return np.sum(out**2)

    num = _numerical_grad_relu_x(loss_relu_sq, x2.copy(), h=1e-5)
    r2 = ReLU()
    out2 = r2.forward(x2)
    analytic = 2 * out2 * (x2 > 0)
    assert np.allclose(num, analytic, rtol=1e-4, atol=1e-4)

    print("relu: all tests passed.")
