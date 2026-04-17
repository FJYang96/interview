import numpy as np


def exact_policy_evaluation(
    P: np.ndarray, R: np.ndarray, policy: np.ndarray, gamma: float = 0.99
) -> np.ndarray:
    """
    Evaluates a policy exactly by solving the linear system of Bellman equations.

    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states).
        R: Reward matrix, shape (num_states, num_actions).
        policy: The stochastic policy to evaluate, shape (num_states, num_actions).
        gamma: Discount factor.

    Returns:
        V: The exact state-value array, shape (num_states,).
    """
    # Your linear algebra implementation here
    # V[s] = sum_a (pi(s, a) * (R(s, a) + gamma * sum_s' P(s'|s, a) * V[s'])
    #      = [sum_a pi(s, a) * R(s, a)] + gamma * sum_s' [sum_a pi(s, a) P(s'|s, a)] * V[s']
    # V    = [R_pi] + gamma * [P_pi] * V
    # (I - gamma * [P_pi]) V = [R_pi]
    P_pi = np.einsum("ijk,ij->ik", P, policy)
    R_pi = (R * policy).sum(1)
    V = np.linalg.solve(np.eye(P.shape[0]) - gamma * P_pi, R_pi)

    return V
