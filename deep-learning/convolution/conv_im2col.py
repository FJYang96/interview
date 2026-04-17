import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def im2col(x, k, stride=1):
    """
    Inputs:
        x: input array of shape (bs, c_in, h_in, w_in)
        k: kernel size (integer)
        stride: integer stride
    Output:
        cols: 2D array of shape (c_in * k * k, bs * h_out * w_out)
    """
    bs, c_in, h_in, w_in = x.shape

    # 1. Extract sliding windows over the spatial dimensions (axis 2 and 3)
    # The resulting shape is (bs, c_in, h_out_unstrided, w_out_unstrided, k, k)
    windows = sliding_window_view(x, window_shape=(k, k), axis=(2, 3))

    # 2. Apply the stride to the spatial output dimensions
    if stride > 1:
        windows = windows[:, :, ::stride, ::stride, :, :]

    # 3. Transpose to group (c_in, k, k) together and (bs, h_out, w_out) together
    # New shape: (c_in, k, k, bs, h_out, w_out)
    windows = windows.transpose(1, 4, 5, 0, 2, 3)

    # 4. Collapse the dimensions into the final 2D matrix
    # Note: Using .reshape() here will force a memory copy, which is required
    # to align the data contiguously for matrix multiplication.
    cols = windows.reshape(c_in * k * k, -1)

    return cols


def forward_im2col(self, x):
    bs, c_in, h_in, w_in = x.shape
    c_out, _, k, _ = self.weights.shape

    # Pad input
    x_padded = self._pad(x)

    # Calculate output dimensions
    h_out = (h_in + 2 * self.padding - k) // self.stride + 1
    w_out = (w_in + 2 * self.padding - k) // self.stride + 1

    # 1. Flatten weights: (c_out, c_in, k, k) -> (c_out, c_in * k * k)
    weights_flat = self.weights.reshape(c_out, -1)

    # 2. Convert padded input to columns: (c_in * k * k, bs * h_out * w_out)
    x_cols = im2col(x_padded, k, self.stride)

    # 3. Perform a single Matrix Multiplication
    # Result shape: (c_out, bs * h_out * w_out)
    out = weights_flat @ x_cols + self.bias

    # 4. Reshape back to image format and transpose batch to the front
    # Shape: (c_out, bs, h_out, w_out) -> (bs, c_out, h_out, w_out)
    y = out.reshape(c_out, bs, h_out, w_out).transpose(1, 0, 2, 3)

    self.x = x
    self.x_padded = x_padded
    self.x_cols = x_cols  # Cache for backprop

    return y


def im2col_1d(x, k, stride=1):
    """
    Inputs:
        x: 1D input array of shape (bs, c_in, L)
        k: kernel size
        stride: integer stride
    Output:
        cols: 2D array of shape (c_in * k, bs * L_out)
    """
    bs, c_in, L = x.shape
    starting_inds = np.arange(0, L - k + 1, stride)[:, None]  # (l_out, 1)
    ind_grid = starting_inds + np.arange(k)[None, :]  # (l_out, k)
    x_cols = x[:, :, ind_grid]  # (bs, c_in, l_out, k)
    cols = x_cols.permute((1, 3, 0, 2)).reshape(c_in * k, -1)

    # bs, c_in, L = x.shape
    # L_out = (L - k) // stride + 1

    # # 1. Create a column vector of starting indices: shape (L_out, 1)
    # start_idx = np.arange(0, L - k + 1, stride)[:, None]

    # # 2. Create a row vector of kernel offsets: shape (k,)
    # offset = np.arange(k)

    # # 3. Broadcast addition to get the full index matrix: shape (L_out, k)
    # idx = start_idx + offset

    # # 4. Apply advanced indexing to extract all windows at once
    # # Resulting shape: (bs, c_in, L_out, k)
    # windows = x[:, :, idx]

    # # 5. Transpose to (c_in, k, bs, L_out) and collapse to 2D
    # cols = windows.transpose(1, 3, 0, 2).reshape(c_in * k, -1)

    return cols
