import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> np.ndarray:
    """
    Computes Generalized Advantage Estimation (GAE) for a single trajectory.

    Args:
        rewards: An array of shape (N,) containing the rewards at each step.
        values: An array of shape (N,) containing the value estimates V(s_t).
        dones: A boolean array of shape (N,) indicating if the episode ended at step t.
        last_value: The value estimate for the state after the final step, V(s_{N}).
                    (0 if the episode actually terminated, otherwise the network's estimate).
        gamma: The discount factor.
        lam: The GAE decay parameter (lambda).

    Returns:
        advantages: An array of shape (N,) containing the computed GAE advantages.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    # Your O(N) backward pass implementation goes here
    # if dones[-1]:
    #     advantages[-1] = rewards[-1] - values[-1]
    # else:
    #     advantages[-1] = rewards[-1] + gamma * last_value - values[-1]

    # for t in range(len(advantages) - 2, -1, -1):
    #     # A_t = \delta_t + gamma * lambda * A_{t+1} if not done at t+1; else A_t = \delta_t
    #     # delta_t = r_t + gamma * V(s_) - V(s)
    #     if dones[t]:
    #         delta_t = rewards[t]
    #         advantages[t] = delta_t
    #     else:
    #         delta_t = rewards[t] + gamma * values[t + 1] - values[t]
    #         advantages[t] = delta_t + gamma * lam * advantages[t + 1]
    last_gae_lam = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0.0
            next_advantage = 0.0
        else:
            next_value = values[t + 1] if t < len(values) else last_value
            next_advantage = last_gae_lam
        delta_t = rewards[t] + gamma * next_value - values[t]
        last_gae_lam = delta_t + gamma * lam * next_advantage
        advantages[t] = last_gae_lam

    return advantages
