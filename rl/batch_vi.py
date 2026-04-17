from typing import Tuple

import numpy as np


def value_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float = 0.99, theta: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Value Iteration to find the optimal value function and policy.

    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states).
        R: Reward matrix, shape (num_states, num_actions).
        gamma: Discount factor.
        theta: Convergence threshold.

    Returns:
        V: The optimal state-value array, shape (num_states,).
        policy: The optimal deterministic policy, shape (num_states,).
                Contains the integer index of the best action for each state.
    """
    num_states, num_actions, _ = P.shape
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    # Your implementation here
    # Value iteration: V[s] = max_a (R[s, a] + gamma * \sum_s' P(s'|s, a) * V[s'])
    while True:
        # 1. compute Q[s, a] = R(s, a) + gamma * sum_s' P(s'|s, a) * V[s'].
        Q_new = R + gamma * np.einsum("ijk,k->ij", P, V)
        # 2. take the max in axis 1.
        V_new = Q_new.max(1)
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new

    # Extract the policy: pi(s) = argmax Q(s, a)
    policy = np.argmax(Q_new, axis=1)

    return V, policy
