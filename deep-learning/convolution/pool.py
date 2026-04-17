import numpy as np


class MaxPool2D:
    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.mask = None  # Useful for routing gradients

    def forward(self, x):
        """
        x shape: (batch_size, channels, height, width)
        Returns shape: (batch_size, channels, out_height, out_width)
        """
        self.x = x

        # Read out the dimensions
        bs, c, h_in, w_in = x.shape
        ps = self.pool_size
        h_out = (h_in - ps) // self.stride + 1
        w_out = (w_in - ps) // self.stride + 1
        y = np.zeros((bs, c, h_out, w_out))
        self.mask = np.zeros((bs, c, h_out, w_out), dtype=int)  # Keep the indices

        # Loop over the output pixels and apply max pooling to each
        b_inds = np.arange(bs)[:, None]
        c_inds = np.arange(c)
        for i in range(h_out):
            for j in range(w_out):
                rf_i = slice(i * self.stride, i * self.stride + ps)
                rf_j = slice(j * self.stride, j * self.stride + ps)
                x_rf = x[:, :, rf_i, rf_j].reshape(bs, c, ps * ps)  # (bs, c, ps * ps)
                max_inds = np.argmax(x_rf, axis=-1)  # (bs, c)
                y[:, :, i, j] = x_rf[b_inds, c_inds, max_inds]
                self.mask[:, :, i, j] = max_inds
        return y

    def backward(self, dL_dout):
        """
        dL_dout shape: (batch_size, channels, out_height, out_width)
        Returns dL_dx shape: (batch_size, channels, height, width)
        """
        # Reading out dimensions
        bs, c, h_out, w_out = dL_dout.shape
        _, _, h_in, w_in = self.x.shape
        dL_dx = np.zeros_like(self.x)
        ps = self.pool_size

        # Loop over the output gradients and route them backwards
        b_inds = np.arange(bs)[:, None]  # (bs, 1)
        c_inds = np.arange(c)[None, :]  # (1, c)
        for i in range(h_out):
            for j in range(w_out):
                max_inds_i = i * self.stride + self.mask[:, :, i, j] // ps  # (bs, c)
                max_inds_j = j * self.stride + self.mask[:, :, i, j] % ps  # (bs, c)
                dL_dx[b_inds, c_inds, max_inds_i, max_inds_j] = dL_dout[
                    :, :, i, j
                ]  # (bs, c)

        return dL_dx


def _maxpool_reference(x, pool_size, stride):
    bs, c, h_in, w_in = x.shape
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    out = np.zeros((bs, c, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            rf_i = slice(i * stride, i * stride + pool_size)
            rf_j = slice(j * stride, j * stride + pool_size)
            out[:, :, i, j] = x[:, :, rf_i, rf_j].max(axis=(2, 3))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 3, 8, 8))
    pool = MaxPool2D(pool_size=2, stride=2)
    out = pool.forward(x)
    ref = _maxpool_reference(x, pool.pool_size, pool.stride)
    assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)
    assert out.shape == (2, 3, 4, 4)

    dL = np.ones_like(out)
    dx = pool.backward(dL)
    assert dx.shape == x.shape
    assert np.isclose(dx.sum(), dL.sum())

    # Numerical gradient: L = sum(maxpool(x))
    x_s = rng.standard_normal((1, 1, 4, 4))

    def loss_pool(arr):
        pl = MaxPool2D(2, stride=2)
        return np.sum(pl.forward(arr))

    num_dx = np.zeros_like(x_s)
    hh = 1e-5
    xs = x_s.copy()
    it = np.nditer(xs, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(xs[idx])
        xs[idx] = old + hh
        fp = loss_pool(xs)
        xs[idx] = old - hh
        fm = loss_pool(xs)
        xs[idx] = old
        num_dx[idx] = (fp - fm) / (2 * hh)
        it.iternext()

    p3 = MaxPool2D(2, stride=2)
    o3 = p3.forward(x_s)
    dx3 = p3.backward(np.ones_like(o3))
    assert np.allclose(dx3, num_dx, rtol=1e-4, atol=1e-4)

    print("maxpool: all tests passed.")
