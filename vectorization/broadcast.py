import numpy as np


###############################################
#                 Problems                    #
###############################################
def pairwise_euclidean(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two sets of vectors.

    Shapes:
        A: (M, D)
        B: (N, D)
        out: (M, N)
    """
    diff = A[:, None, :] - B[None, :, :]  # (M, N, D)
    return np.sqrt((diff**2).sum(-1))


def normalize_images(images, means, stds):
    """
    Normalizes a batch of channel-first images.
    Formula per channel: (image - mean) / std

    Shapes:
        images: (B, C, H, W)
        means: (C,)
        stds: (C,)
        out: (B, C, H, W)
    """
    return (images - means[None, :, None, None]) / stds[None, :, None, None]


def batch_weighted_outer(V, W):
    """
    Computes a batch of weighted outer products.

    Shapes:
        V: (B, D)
        W: (D,)
        Out: (B, D, D)

    Output[b, i, j] = V[b, i] * W[j] * V[b, j]
    """
    return V[:, :, None] * V[:, None, :] * W[None, None, :]
    # return np.einsum("bi,bj,j->bij", V, V, W)


def sparse_to_one_hot(labels, num_classes):
    """
    Converts a 1D array of sparse integer labels to a 2D one-hot boolean mask.
    Must be implemented via broadcasting a comparison.

    Shapes:
        labels: (B,)
        num_classes: int
        out: (B, num_classes)
    """
    B = len(labels)
    one_hot = np.zeros((B, num_classes), dtype=bool)
    one_hot[np.arange(B), labels] = 1  # indices are (B,) and (B,) -> broadcasts to (B,)
    return one_hot


def batch_affine_transform(points, A, b):
    """
    points: (B, N, D), N points per batch
    A: (B, D, D)
    b: (B, D)

    Returns: (B, N, D)
    """
    Ap_product = (
        A[:, None, :, :] * points[:, :, None, :]
    )  # (B, N, D, D), multiply on axis=2
    return Ap_product.sum(3) + b[:, None, :]


def masked_softmax(logits, mask):
    """
    logits: (B, N, K)
    mask: (B, N, K), binary (0 or 1)

    Returns: (B, N, K)
    """
    shifted_logits = logits.copy() - logits.max(-1, keepdims=True)
    shifted_logits[~mask.astype(bool)] = -np.inf
    logits_exp = np.exp(shifted_logits)
    sm = logits_exp / (logits_exp.sum(-1, keepdims=True) + 1e-8)
    return sm


def batched_cosine_similarity(X):
    """
    X: (B, N, D)

    Returns: (B, N, N) similarity matrix per batch

    Out[b, i, j] = X[b, i] @ X[b, j] / norm(X[b, i]) norm(X[b, j])
    """
    dots = (X[:, :, None, :] * X[:, None, :, :]).sum(
        -1
    )  # (B, N, 1, D) * (B, 1, N, D) -> (B, N, N, D) -> (B, N, N)
    norms = np.linalg.norm(X, axis=-1)  # (B, N)
    ret = dots / (norms[:, :, None] * norms[:, None, :] + 1e-9)
    return ret


def gaussian_kernel(X, Y, sigma):
    """
    X: (B, N, D)
    Y: (B, M, D)
    sigma: scalar

    Returns: (B, N, M)

    Out[b, i, j] = exp(-(X[b, i] - Y[b, j])^2 / (2 sigma^2))
    """
    dist = np.sum(
        (X[:, :, None, :] - Y[:, None, :, :]) ** 2, axis=-1
    )  # (B, N, 1, D) - (B, 1, M, D) -> (B, N, M, D) -> (B, N, M)
    return np.exp(-dist / (2 * sigma**2))


def create_ragged_mask(lengths, max_len):
    """
    Creates a boolean mask for ragged sequences.

    Args:
        lengths: np.ndarray of shape (B,) containing integer lengths.
        max_len: int, the maximum possible sequence length.

    Returns:
        np.ndarray of shape (B, max_len) of dtype bool.
    """
    mask = np.arange(max_len)[None, :] < lengths[:, None]  # (1, L) < (B, 1) -> (B, L)
    return mask


def batched_bilinear(X, W, Y):
    """
    Computes a batched bilinear form of x^\top W y

    Args:
        X: np.ndarray of shape (B, N, D)
        W: np.ndarray of shape (D, D)
        Y: np.ndarray of shape (B, M, D)

    Returns:
        np.ndarray of shape (B, N, M)
    """
    return np.einsum("bni,ij,bmj->bnm", X, W, Y)


def scaled_dot_product_attention(Q, K, V, mask):
    """
    Computes multi-head attention.

    Args:
        Q: np.ndarray of shape (B, H, N, D)
        K: np.ndarray of shape (B, H, M, D)
        V: np.ndarray of shape (B, H, M, D)
        mask: boolean np.ndarray of shape (B, 1, N, M). False means do not attend.

    Returns:
        np.ndarray of shape (B, H, N, D)
    """
    # Your batched matmul, masking, and softmax code here
    QK = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(Q.shape[-1])  # (B, H, N, M)
    shifted_qk = QK - QK.max(-1, keepdims=True)  # (B, H, N, M)
    masked_qk = np.where(mask, shifted_qk, -np.inf)
    exp_att = np.exp(masked_qk)
    sm = exp_att / exp_att.sum(-1, keepdims=True)  # (B, H, N, M)
    return sm @ V


def nearest_neighbor_idx(X, Y):
    """
    X: (N, D)
    Y: (M, D)

    Returns:
        idx: (N,) where idx[i] = argmin_j ||X[i] - Y[j]||^2
    """
    dist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    return np.argmin(dist, axis=-1)


def topk_per_row(X, k):
    """
    X: (N, D)
    k: int

    Returns:
        values: (N, k) top-k values per row (sorted descending)
    """
    ret = np.partition(X, -k, axis=-1)[:, -k:]
    ret = np.sort(ret, axis=1)[:, ::-1]
    return ret


def grouped_mean(X, group_ids, num_groups):
    """
    Compute mean per group given group indices.

    X: (N, D)
    group_ids: (N,) integers in [0, num_groups)
    num_groups: int

    Returns:
        (num_groups, D)
    """
    N, D = X.shape

    X_group = X[None, :, :]  # (1, N, D)
    in_group = (
        np.arange(num_groups)[:, None, None] == group_ids[None, :, None]
    )  # (G, N, 1)
    prod = X_group * in_group  # (G, N, D)
    return np.where(in_group.sum(1) == 0, 0, prod.sum(1) / (in_group.sum(1) + 1e-9))


def segmented_max(X, segment_ids, num_segments):
    """
    Compute max per segment (like grouped max).
    If nothing in segment, return -np.inf.

    X: (N,)
    segment_ids: (N,)
    num_segments: int

    Returns:
        (num_segments,)
    """
    in_segment = np.arange(num_segments)[:, None] == segment_ids  # (S, N)
    X_masked = in_segment * X  # (S, N)
    count_per_seg = in_segment.sum(1)  # (S,)
    ret = np.where(count_per_seg == 0, -np.inf, X_masked.max(1))
    return ret


def masked_argmax(X, mask):
    """
    For each row, pick argmax only over valid entries.
    If entire row masked, return -1

    X: (B, N)
    mask: (B, N) boolean

    Returns:
        (B,) indices
    """
    ret = np.argmax(np.where(mask, X, -np.inf), axis=-1)
    return np.where(mask.sum(1) == 0, -1, ret)


###############################################
#                 Tests                       #
###############################################
def test_pairwise_euclidean():
    M, N, D = 13, 7, 5  # Distinct dimensions to prevent silent shape-matching bugs
    A = np.random.rand(M, D)
    B = np.random.rand(N, D)

    distances = pairwise_euclidean(A, B)

    assert distances is not None, "Function returned None"
    assert distances.shape == (M, N), f"Expected shape {(M, N)}, got {distances.shape}"

    # Ground truth: explicit loops
    expected = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            expected[i, j] = np.sqrt(np.sum((A[i] - B[j]) ** 2))

    np.testing.assert_allclose(
        distances, expected, err_msg="Pairwise distance calculation is incorrect"
    )


def test_normalize_images():
    B, C, H, W = 5, 3, 11, 13  # C != H != W ensures broadcasting hits the correct axis
    images = np.random.rand(B, C, H, W) * 255
    means = np.random.rand(C) * 255
    stds = np.random.rand(C) * 255 + 1e-5  # prevent division by zero

    normalized = normalize_images(images, means, stds)

    assert normalized is not None, "Function returned None"
    assert normalized.shape == (B, C, H, W), (
        f"Expected shape {(B, C, H, W)}, got {normalized.shape}"
    )

    # Ground truth: explicit loops
    expected = np.zeros_like(images)
    for b in range(B):
        for c in range(C):
            expected[b, c, :, :] = (images[b, c, :, :] - means[c]) / stds[c]

    np.testing.assert_allclose(
        normalized, expected, err_msg="Image normalization is incorrect"
    )


def test_batch_weighted_outer():
    B, N = 7, 5
    V = np.random.rand(B, N)
    W = np.random.rand(N)

    output = batch_weighted_outer(V, W)

    assert output is not None, "Function returned None"
    assert output.shape == (B, N, N), f"Expected shape {(B, N, N)}, got {output.shape}"

    # Ground truth: explicit loops
    expected = np.zeros((B, N, N))
    for b in range(B):
        for i in range(N):
            for j in range(N):
                expected[b, i, j] = V[b, i] * W[j] * V[b, j]

    np.testing.assert_allclose(
        output, expected, err_msg="Batch weighted outer product is incorrect"
    )


def test_sparse_to_one_hot():
    B = 17
    C = 6
    labels = np.random.randint(0, C, size=(B,))

    one_hot = sparse_to_one_hot(labels, C)

    assert one_hot is not None, "Function returned None"
    assert one_hot.shape == (B, C), f"Expected shape {(B, C)}, got {one_hot.shape}"
    assert one_hot.dtype == bool, f"Expected boolean array, got {one_hot.dtype}"
    assert np.all(one_hot.sum(axis=1) == 1), (
        "Each row must contain exactly one True value"
    )

    # Ground truth: explicit indexing
    expected = np.zeros((B, C), dtype=bool)
    for b in range(B):
        expected[b, labels[b]] = True

    np.testing.assert_array_equal(
        one_hot, expected, err_msg="One-hot mask generation is incorrect"
    )


def test_batch_affine_transform():
    def reference(points, A, b):
        B, N, D = points.shape
        out = np.zeros_like(points)
        for b_idx in range(B):
            for i in range(N):
                out[b_idx, i] = A[b_idx] @ points[b_idx, i] + b[b_idx]
        return out

    # Basic
    B, N, D = 2, 3, 2
    points = np.arange(B * N * D).reshape(B, N, D)
    A = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    b = np.array([[1, 1], [0, -1]])

    out = batch_affine_transform(points, A, b)
    assert out.shape == (B, N, D)
    assert np.allclose(out, reference(points, A, b))

    # Random
    np.random.seed(1)
    B, N, D = 4, 5, 3
    points = np.random.randn(B, N, D)
    A = np.random.randn(B, D, D)
    b = np.random.randn(B, D)

    out = batch_affine_transform(points, A, b)
    assert np.allclose(out, reference(points, A, b), atol=1e-6)

    # Edge: N=1
    points = np.random.randn(B, 1, D)
    out = batch_affine_transform(points, A, b)
    assert out.shape == (B, 1, D)


def test_masked_softmax():
    def reference(logits, mask):
        masked_logits = np.where(mask == 0, logits, -np.inf)
        max_logits = np.max(masked_logits, axis=-1, keepdims=True)

        # 2. Protect against fully masked rows where max_logits is -inf.
        # Subtracting -inf from logits would cause +inf. We temporarily replace -inf with 0.0.
        # The actual value doesn't matter because we will mask these out again in step 4.
        safe_max_logits = np.where(np.isinf(max_logits), 0.0, max_logits)

        # 3. Shift logits for numerical stability.
        shifted_logits = logits - safe_max_logits

        # 4. CRITICAL FIX: Force masked elements to -np.inf BEFORE exponentiation.
        # This prevents exp() from overflowing on masked elements and generating Infs.
        # np.exp(-np.inf) yields exactly 0.0, naturally applying our mask.
        shifted_logits = np.where(mask == 1, shifted_logits, -np.inf)

        # 5. Exponentiate. Masked items cleanly become 0.0. No NaN-generating math occurs.
        exp_logits = np.exp(shifted_logits)

        # 6. Sum the exponentials along the last axis.
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)

        # 7. Protect against division by zero for fully masked rows.
        sum_exp_safe = np.where(sum_exp == 0.0, 1.0, sum_exp)

        # 8. Normalize. Fully masked rows will safely compute 0.0 / 1.0 = 0.0.
        probs = exp_logits / sum_exp_safe

        return probs

    np.random.seed(0)

    logits = np.random.randn(2, 3, 4)
    mask = np.array(
        [
            [[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]],
            [[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
        ]
    )

    out = masked_softmax(logits, mask)
    ref = reference(logits, mask)

    assert out.shape == logits.shape
    assert np.allclose(out, ref, atol=1e-6)

    # Mask correctness
    assert np.all(out[mask == 0] == 0)

    # Row sum = 1 for valid rows
    row_sums = out.sum(axis=-1)
    valid_rows = mask.sum(axis=-1) > 0
    assert np.allclose(row_sums[valid_rows], 1)

    # Edge: fully masked
    logits = np.random.randn(1, 2, 3)
    mask = np.zeros_like(logits)
    out = masked_softmax(logits, mask)
    assert np.all(out == 0)


def test_batched_cosine_similarity():
    def reference(X):
        B, N, D = X.shape
        out = np.zeros((B, N, N))
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    xi = X[b, i]
                    xj = X[b, j]
                    denom = np.linalg.norm(xi) * np.linalg.norm(xj)
                    if denom == 0:
                        out[b, i, j] = 0
                    else:
                        out[b, i, j] = np.dot(xi, xj) / denom
        return out

    np.random.seed(1)
    X = np.random.randn(2, 4, 3)

    out = batched_cosine_similarity(X)
    ref = reference(X)

    assert out.shape == (2, 4, 4)
    assert np.allclose(out, ref, atol=1e-6)

    # Symmetry
    assert np.allclose(out, np.transpose(out, (0, 2, 1)))

    # Diagonal = 1
    diag = np.diagonal(out, axis1=1, axis2=2)
    assert np.allclose(diag, 1)

    # Edge: zero vectors
    X[:, 0] = 0
    out = batched_cosine_similarity(X)
    assert np.all(out[:, 0, :] == 0)
    assert np.all(out[:, :, 0] == 0)


def test_gaussian_kernel():
    def reference(X, Y, sigma):
        B, N, D = X.shape
        M = Y.shape[1]
        out = np.zeros((B, N, M))
        for b in range(B):
            for i in range(N):
                for j in range(M):
                    diff = X[b, i] - Y[b, j]
                    sq = np.sum(diff**2)
                    out[b, i, j] = np.exp(-sq / (2 * sigma**2))
        return out

    np.random.seed(0)

    X = np.random.randn(2, 3, 2)
    Y = np.random.randn(2, 4, 2)
    sigma = 1.5

    out = gaussian_kernel(X, Y, sigma)
    ref = reference(X, Y, sigma)

    assert out.shape == (2, 3, 4)
    assert np.allclose(out, ref, atol=1e-6)

    # Bounds
    assert np.all(out > 0)
    assert np.all(out <= 1)

    # Self-kernel diagonal = 1
    out_self = gaussian_kernel(X, X, sigma)
    diag = np.diagonal(out_self, axis1=1, axis2=2)
    assert np.allclose(diag, 1)

    # Smaller sigma => sharper kernel
    out_small = gaussian_kernel(X, Y, 0.1)
    assert np.all(out_small <= out + 1e-6)


def test_create_ragged_mask():
    B, max_len = 13, 11  # Distinct primes

    # Generate random sequence lengths between 0 and max_len
    lengths = np.random.randint(0, max_len + 1, size=(B,))

    mask = create_ragged_mask(lengths, max_len)

    assert mask is not None, "Function returned None"
    assert mask.shape == (B, max_len), (
        f"Expected shape {(B, max_len)}, got {mask.shape}"
    )
    assert mask.dtype == bool, f"Expected boolean array, got {mask.dtype}"

    # Ground truth: slow explicit loops
    expected = np.zeros((B, max_len), dtype=bool)
    for b in range(B):
        for i in range(max_len):
            if i < lengths[b]:
                expected[b, i] = True

    np.testing.assert_array_equal(
        mask, expected, err_msg="Ragged mask generation is incorrect"
    )


def test_batched_bilinear():
    B, N, M, D = 5, 7, 11, 3  # Distinct primes

    X = np.random.randn(B, N, D)
    W = np.random.randn(D, D)
    Y = np.random.randn(B, M, D)

    output = batched_bilinear(X, W, Y)

    assert output is not None, "Function returned None"
    assert output.shape == (B, N, M), f"Expected shape {(B, N, M)}, got {output.shape}"

    # Ground truth: slow explicit loops
    expected = np.zeros((B, N, M))
    for b in range(B):
        for n in range(N):
            for m in range(M):
                # Isolate the vectors and compute x^T W y
                x_vec = X[b, n]
                y_vec = Y[b, m]
                expected[b, n, m] = x_vec @ W @ y_vec

    np.testing.assert_allclose(
        output, expected, atol=1e-7, err_msg="Batched bilinear form is incorrect"
    )


def test_scaled_dot_product_attention():
    B, H, N, M, D = 3, 5, 7, 11, 13

    Q = np.random.randn(B, H, N, D)
    K = np.random.randn(B, H, M, D)
    V = np.random.randn(B, H, M, D)

    # Create a random boolean mask of shape (B, 1, N, M)
    mask = np.random.randint(0, 2, size=(B, 1, N, M)).astype(bool)

    # Force the first key to always be valid for all queries to avoid division by zero
    mask[:, :, :, 0] = True

    output = scaled_dot_product_attention(Q, K, V, mask)

    assert output is not None, "Function returned None"
    assert output.shape == (B, H, N, D), (
        f"Expected shape {(B, H, N, D)}, got {output.shape}"
    )

    # Ground truth: slow explicit loops
    expected = np.zeros((B, H, N, D))
    scale = np.sqrt(D)

    for b in range(B):
        for h in range(H):
            for n in range(N):
                # 1. Compute unscaled dot products for this query against all keys
                scores = np.zeros(M)
                for m in range(M):
                    scores[m] = np.dot(Q[b, h, n], K[b, h, m])

                # 2. Scale
                scores = scores / scale

                # 3. Apply masking
                valid_keys = mask[b, 0, n, :]
                masked_scores = np.where(valid_keys, scores, -np.inf)

                # 4. Standard Softmax on the isolated row
                shifted = masked_scores - np.max(masked_scores)
                exps = np.exp(shifted)
                probs = exps / np.sum(exps)

                # 5. Multiply by Values and sum
                out_vec = np.zeros(D)
                for m in range(M):
                    out_vec += probs[m] * V[b, h, m]

                expected[b, h, n] = out_vec

    np.testing.assert_allclose(
        output, expected, atol=1e-6, err_msg="Attention logic is incorrect"
    )


def test_nearest_neighbor_idx():
    def reference(X, Y):
        N = X.shape[0]
        idx = np.zeros(N, dtype=int)
        for i in range(N):
            dists = np.sum((Y - X[i]) ** 2, axis=1)
            idx[i] = np.argmin(dists)
        return idx

    np.random.seed(0)

    X = np.random.randn(10, 3)
    Y = np.random.randn(15, 3)

    out = nearest_neighbor_idx(X, Y)
    ref = reference(X, Y)

    assert out.shape == (10,)
    assert np.all(out == ref)

    # Edge: identical sets
    out = nearest_neighbor_idx(X, X)
    assert np.all(out == np.arange(len(X)))


def test_topk_per_row():
    def reference(X, k):
        return np.sort(X, axis=1)[:, -k:][:, ::-1]

    np.random.seed(1)

    X = np.random.randn(8, 10)
    k = 3

    out = topk_per_row(X, k)
    ref = reference(X, k)

    assert out.shape == (8, k)
    assert np.allclose(out, ref)

    # Edge: k = 1
    out = topk_per_row(X, 1)
    assert out.shape == (8, 1)
    assert np.allclose(out[:, 0], np.max(X, axis=1))

    # Edge: duplicates
    X = np.ones((5, 6))
    out = topk_per_row(X, 2)
    assert np.all(out == 1)


def test_grouped_mean():
    def reference(X, group_ids, G):
        out = np.zeros((G, X.shape[1]))
        counts = np.zeros(G)
        for i in range(len(X)):
            g = group_ids[i]
            out[g] += X[i]
            counts[g] += 1
        for g in range(G):
            if counts[g] > 0:
                out[g] /= counts[g]
        return out

    np.random.seed(0)

    X = np.random.randn(20, 4)
    group_ids = np.random.randint(0, 5, size=20)

    out = grouped_mean(X, group_ids, 5)
    ref = reference(X, group_ids, 5)

    assert out.shape == (5, 4)
    assert np.allclose(out, ref)

    # Edge: empty group
    group_ids[:] = 0
    out = grouped_mean(X, group_ids, 3)
    assert np.all(out[1:] == 0)


def test_segmented_max():
    def reference(X, seg, K):
        out = np.full(K, -np.inf)
        for i in range(len(X)):
            out[seg[i]] = max(out[seg[i]], X[i])
        return out

    np.random.seed(0)

    X = np.random.randn(20)
    seg = np.random.randint(0, 6, size=20)

    out = segmented_max(X, seg, 6)
    ref = reference(X, seg, 6)

    assert out.shape == (6,)
    assert np.allclose(out, ref)

    # Edge: single segment
    seg[:] = 0
    out = segmented_max(X, seg, 3)
    assert out[0] == np.max(X)
    assert np.all(np.isneginf(out[1:]))


def test_masked_argmax():
    def reference(X, mask):
        B, N = X.shape
        out = np.zeros(B, dtype=int)
        for b in range(B):
            valid = np.where(mask[b])[0]
            if len(valid) == 0:
                out[b] = -1
            else:
                vals = X[b, valid]
                out[b] = valid[np.argmax(vals)]
        return out

    np.random.seed(0)

    X = np.random.randn(5, 6)
    mask = np.random.rand(5, 6) > 0.3

    out = masked_argmax(X, mask)
    ref = reference(X, mask)

    assert out.shape == (5,)
    assert np.all(out == ref)

    # Edge: fully masked row
    mask[0] = False
    out = masked_argmax(X, mask)
    assert out[0] == -1


test_pairwise_euclidean()
test_normalize_images()
test_batch_weighted_outer()
test_sparse_to_one_hot()
test_batch_affine_transform()
test_masked_softmax()
test_batched_cosine_similarity()
test_gaussian_kernel()
test_create_ragged_mask()
test_batched_bilinear()
test_scaled_dot_product_attention()
test_nearest_neighbor_idx()
test_topk_per_row()
test_grouped_mean()
test_segmented_max()
test_masked_argmax()
