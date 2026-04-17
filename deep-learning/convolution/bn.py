import numpy as np


class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (Scale and Shift)
        # Shape: (1, num_features, 1, 1) for broadcasting
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))

        # Exponential moving averages for inference
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        # Cache for backward pass
        self.cache = None

    def forward(self, x, train=True):
        """
        x shape: (batch_size, num_features, height, width)
        Returns shape: (batch_size, num_features, height, width)
        """
        if train:
            mu = np.mean(x, axis=(0, 2, 3), keepdims=True)  # (1, F, 1, 1)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)  # (1, F, 1, 1)

            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mu
            self.running_var = (1 - m) * self.running_mean + m * var
        else:
            mu = self.running_mean
            var = self.running_var

        x_hat = (x - mu) / np.sqrt(var + self.eps)
        y = self.gamma * x_hat + self.beta

        self.cache = (x, mu, var, x_hat)

        return y

    def backward(self, dL_dout, lr):
        """
        dL_dout shape: (batch_size, num_features, height, width)
        Returns dL_dx shape: (batch_size, num_features, height, width)
        """
        if self.cache is None:
            raise Exception("Attempt to backprop without forward pass.")

        # Unpack cache
        x, batch_mean, batch_var, x_norm = self.cache

        dL_dgamma = (dL_dout * x_norm).sum((0, 2, 3), keepdims=True)
        dL_dbeta = dL_dout.sum((0, 2, 3), keepdims=True)

        m = x.shape[0] * x.shape[2] * x.shape[3]
        std_inv = 1.0 / np.sqrt(batch_var + self.eps)
        dxhat = dL_dout * self.gamma
        dL_dx = (std_inv / m) * (
            m * dxhat
            - dxhat.sum(axis=(0, 2, 3), keepdims=True)
            - x_norm * (dxhat * x_norm).sum(axis=(0, 2, 3), keepdims=True)
        )
        self.gamma -= lr * dL_dgamma
        self.beta -= lr * dL_dbeta

        return dL_dx


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    bs, c, h, w = 4, 3, 5, 5
    x = rng.standard_normal((bs, c, h, w))
    bn = BatchNorm2D(c, eps=1e-5, momentum=0.1)
    bn.gamma = np.ones_like(bn.gamma)
    bn.beta = np.zeros_like(bn.beta)

    y = bn.forward(x, train=True)
    assert y.shape == x.shape
    # Per-channel (over batch and spatial) mean ~ 0, variance ~ 1 for normalized activations
    for ch in range(c):
        slice_y = y[:, ch, :, :].ravel()
        assert np.abs(slice_y.mean()) < 1e-6
        assert np.isclose(slice_y.var(), 1.0, rtol=1e-5, atol=1e-5)

    # Inference path uses running stats (after one train forward, running is blended)
    bn2 = BatchNorm2D(c, eps=1e-5, momentum=0.5)
    bn2.gamma = np.ones_like(bn2.gamma)
    bn2.beta = np.zeros_like(bn2.beta)
    _ = bn2.forward(x, train=True)
    y_eval = bn2.forward(x, train=False)
    assert y_eval.shape == x.shape

    # Backward vs numerical gradient: L = sum(y^2), train mode
    bn3 = BatchNorm2D(2, eps=1e-5, momentum=0.1)
    bn3.gamma = rng.standard_normal((1, 2, 1, 1)) * 0.5 + 1.0
    bn3.beta = rng.standard_normal((1, 2, 1, 1)) * 0.1
    x_small = rng.standard_normal((2, 2, 3, 3))
    y_small = bn3.forward(x_small, train=True)
    dL = 2 * y_small
    g0_gamma, g0_beta = bn3.gamma.copy(), bn3.beta.copy()
    dx = bn3.backward(dL, lr=0.0)
    assert np.allclose(bn3.gamma, g0_gamma) and np.allclose(bn3.beta, g0_beta)
    assert dx.shape == x_small.shape

    def loss_bn(arr):
        b = BatchNorm2D(2, eps=1e-5, momentum=0.1)
        b.gamma = bn3.gamma.copy()
        b.beta = bn3.beta.copy()
        out = b.forward(arr, train=True)
        return np.sum(out**2)

    num_dx = np.zeros_like(x_small)
    hh = 1e-4
    x_work = x_small.copy()
    it = np.nditer(x_work, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(x_work[idx])
        x_work[idx] = old + hh
        fp = loss_bn(x_work)
        x_work[idx] = old - hh
        fm = loss_bn(x_work)
        x_work[idx] = old
        num_dx[idx] = (fp - fm) / (2 * hh)
        it.iternext()

    bn4 = BatchNorm2D(2, eps=1e-5, momentum=0.1)
    bn4.gamma = bn3.gamma.copy()
    bn4.beta = bn3.beta.copy()
    y4 = bn4.forward(x_small, train=True)
    dx4 = bn4.backward(2 * y4, lr=0.0)
    assert np.allclose(dx4, num_dx, rtol=2e-3, atol=2e-3)

    print("batchnorm: all tests passed.")
