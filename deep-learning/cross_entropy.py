import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.cache = None

    def forward(self, logits, y):
        """
        logits shape: (batch_size, num_classes)
        y shape: (batch_size,) - Integer class labels
        Returns: Scalar loss
        """
        bs, _ = logits.shape

        # Find the probability
        logits_normalized = logits - logits.max(1, keepdims=True)
        logits_exp = np.exp(logits_normalized)
        prob = logits_exp / logits_exp.sum(1, keepdims=True)

        # Find the loss
        prob_true = prob[np.arange(bs), y]  # (bs,)
        loss = -np.log(prob_true + 1e-15).sum() / bs

        self.cache = prob, y

        return loss

    def backward(self):
        """
        Returns dL_dz shape: (batch_size, num_classes)
        """
        if self.cache is None:
            raise Exception("Attempt to backprop without forward pass.")

        probs, y = self.cache
        bs = probs.shape[0]
        dL_dz = probs
        dL_dz[np.arange(bs), y] -= 1
        dL_dz /= bs

        return dL_dz


def _numerical_grad_logits(loss_fn, logits, y, h=1e-5):
    """Scalar loss w.r.t. logits via central differences."""
    grad = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            orig = logits[i, j]
            logits[i, j] = orig + h
            lp = loss_fn(logits, y)
            logits[i, j] = orig - h
            lm = loss_fn(logits, y)
            logits[i, j] = orig
            grad[i, j] = (lp - lm) / (2 * h)
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Forward: loss matches -mean(log p[y]) for computed softmax
    logits = rng.standard_normal((4, 5))
    y = np.array([0, 2, 1, 4])
    ce = CrossEntropyLoss()
    loss = ce.forward(logits, y)
    logits_max = np.max(logits, axis=1, keepdims=True)
    shifted = logits - logits_max
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    expected = -np.mean(np.log(probs[np.arange(4), y] + 1e-15))
    assert np.isclose(loss, expected, rtol=1e-5, atol=1e-7)

    # Backward: analytic gradient vs numerical (small problem)
    logits_small = rng.standard_normal((2, 3))
    y_small = np.array([1, 0])
    ce2 = CrossEntropyLoss()
    ce2.forward(logits_small, y_small)
    analytic = ce2.backward()

    def loss_only(z, labels):
        c = CrossEntropyLoss()
        return c.forward(z, labels)

    numeric = _numerical_grad_logits(loss_only, logits_small.copy(), y_small)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-4)

    # Known toy: uniform logits -> loss log(num_classes); grad rows sum to 0
    k = 3
    logits_u = np.zeros((2, k))
    y_u = np.array([0, 1])
    ce3 = CrossEntropyLoss()
    loss_u = ce3.forward(logits_u, y_u)
    assert np.isclose(loss_u, np.log(k), rtol=1e-6)
    g = ce3.backward()
    assert np.allclose(g.sum(axis=1), 0, atol=1e-7)

    print("cross_entropy: all tests passed.")
