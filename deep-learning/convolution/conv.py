import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.bias = np.random.randn(out_channels, 1)
        self.stride = stride
        self.padding = padding

        # placeholder for backprop cache
        self.x = None
        self.x_padded = None

    def _pad(self, x):
        """
        Input:
            x: input of shape (bs, c_in, h_in, w_in)
        Output:
            x_padded: padded input (bs, c_in, h_in + 2 * p, w_in + 2 * p)
        """
        if self.padding == 0:
            return x
        bs, c_in, h_in, w_in = x.shape
        p = self.padding
        x_padded = np.zeros((bs, c_in, h_in + 2 * p, w_in + 2 * p))
        x_padded[:, :, p:-p, p:-p] = x
        return x_padded

    def forward(self, x):
        """
        Input
            x: input of shape (bs, c_in, h_in, w_in)
        Return
            y: output of shape (bs, c_out, h_out, w_out)
        """
        # Compute the output dimensions
        bs, c_in, h_in, w_in = x.shape
        c_out, _, k, _ = self.weights.shape
        p = self.padding
        h_out = (h_in + 2 * p - k) // self.stride + 1
        w_out = (w_in + 2 * p - k) // self.stride + 1
        y = np.zeros((bs, c_out, h_out, w_out))

        # Pad the inputs
        x_padded = self._pad(x)

        # Find the output pixels one by one
        for i in range(h_out):
            for j in range(w_out):
                rf_i = slice(i * self.stride, i * self.stride + k)
                rf_j = slice(j * self.stride, j * self.stride + k)
                x_rf = x_padded[:, :, rf_i, rf_j]  # (bs, c_in, k, k)
                y[:, :, i, j] = np.einsum(
                    "bcmn,dcmn->bd", x_rf, self.weights
                ) + self.bias.reshape(1, c_out)  # (bs, c_out)

        # cache the padded inputs for backprop
        self.x = x
        self.x_padded = x_padded
        return y

    def backward(self, dL_dout, lr):
        """
        Inputs:
            dL_dout: (bs, c_out, h_out, w_out)
            lr: float
        Output:
            dL_dx: (bs, c_in, h_in, w_in)
        """
        # Read out the dimensions
        bs, c_out, h_out, w_out = dL_dout.shape
        _, c_in, h_in_p, w_in_p = self.x_padded.shape
        k = self.weights.shape[2]

        # Create the gradient tensors
        dL_dxp = np.zeros_like(
            self.x_padded
        )  # Compute the gradient for the padded x first
        dL_dW = np.zeros_like(self.weights)
        dL_db = np.zeros_like(self.bias)

        # Loop over the output pixels
        for i in range(h_out):
            for j in range(w_out):
                rf_i = slice(i * self.stride, i * self.stride + k)
                rf_j = slice(j * self.stride, j * self.stride + k)
                x_rf = self.x_padded[:, :, rf_i, rf_j]  # (bs, c_in, k, k)
                dL_dout_ij = dL_dout[:, :, i, j]  # (bs, c_out)
                dL_dW += np.einsum(
                    "bd,bcmn->dcmn", dL_dout_ij, x_rf
                )  # (c_out, c_in, k, k)
                dL_db += np.sum(dL_dout, axis=(0, 2, 3)).reshape(c_out, 1)  # (c_out, 1)
                dL_dxp[:, :, rf_i, rf_j] += np.einsum(
                    "bd,dcmn->bcmn", dL_dout_ij, self.weights
                )  # (bs, c_in, k, k)

        # Extract the gradient from the padded input
        if self.padding == 0:
            dL_dx = dL_dxp
        else:
            p = self.padding
            dL_dx = dL_dxp[:, :, p:-p, p:-p]

        return dL_dx


def _conv_reference_forward(x, weights, bias, stride, padding):
    """Slow reference conv for shape (bs, c_in, h, w) — used only in tests."""
    bs, c_in, h_in, w_in = x.shape
    c_out, _, k, _ = weights.shape
    if padding:
        p = padding
        xp = np.zeros((bs, c_in, h_in + 2 * p, w_in + 2 * p))
        xp[:, :, p : p + h_in, p : p + w_in] = x
    else:
        xp = x
    h_out = (h_in + 2 * padding - k) // stride + 1
    w_out = (w_in + 2 * padding - k) // stride + 1
    out = np.zeros((bs, c_out, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            rf_i = slice(i * stride, i * stride + k)
            rf_j = slice(j * stride, j * stride + k)
            x_rf = xp[:, :, rf_i, rf_j]
            out[:, :, i, j] = np.einsum("bcij, dcij -> bd", x_rf, weights)
    out += bias.reshape(1, c_out, 1, 1)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    bs, cin, cout, k, h_in, w_in = 1, 2, 2, 3, 6, 5
    x = rng.standard_normal((bs, cin, h_in, w_in))

    def make_layer():
        c = Conv2D(cin, cout, k, stride=1, padding=1)
        c.weights = rng.standard_normal((cout, cin, k, k))
        c.bias = rng.standard_normal((cout, 1))
        return c

    layer = make_layer()
    out = layer.forward(x)
    ref = _conv_reference_forward(
        x, layer.weights, layer.bias, layer.stride, layer.padding
    )
    assert np.allclose(out, ref, rtol=1e-10, atol=1e-10)

    # Backward vs numeric on input: L = sum(out^2)
    x_fix = rng.standard_normal((bs, cin, h_in, w_in))
    L_fix = make_layer()
    w_fix = L_fix.weights.copy()
    b_fix = L_fix.bias.copy()

    o = L_fix.forward(x_fix)
    dL = 2 * o
    W0, b0 = L_fix.weights.copy(), L_fix.bias.copy()
    dx = L_fix.backward(dL, lr=0.0)
    assert np.allclose(L_fix.weights, W0) and np.allclose(L_fix.bias, b0)

    def loss_x(arr):
        L = Conv2D(cin, cout, k, stride=1, padding=1)
        L.weights = w_fix.copy()
        L.bias = b_fix.copy()
        return np.sum(L.forward(arr) ** 2)

    num_dx = np.zeros_like(x_fix)
    hh = 1e-4
    it = np.nditer(x_fix, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(x_fix[idx])
        x_fix[idx] = old + hh
        fp = loss_x(x_fix)
        x_fix[idx] = old - hh
        fm = loss_x(x_fix)
        x_fix[idx] = old
        num_dx[idx] = (fp - fm) / (2 * hh)
        it.iternext()

    L_fix2 = Conv2D(cin, cout, k, stride=1, padding=1)
    L_fix2.weights = w_fix.copy()
    L_fix2.bias = b_fix.copy()
    o2 = L_fix2.forward(x_fix)
    dx_analytic = L_fix2.backward(2 * o2, lr=0.0)
    assert np.allclose(dx_analytic, num_dx, rtol=2e-3, atol=2e-3)

    print("conv: all tests passed.")
