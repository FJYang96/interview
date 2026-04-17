import numpy as np


def vectorized_policy_evaluation(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-6,
) -> np.ndarray:
    """
    Iteratively evaluates a policy using vectorized operations.

    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states).
           P[s, a, s'] is the probability of transitioning to s' from s given a.
        R: Reward matrix, shape (num_states, num_actions).
           R[s, a] is the expected reward for taking action a in state s.
        policy: The stochastic policy to evaluate, shape (num_states, num_actions).
           policy[s, a] is the probability of taking action a in state s.
        gamma: Discount factor.
        theta: Convergence threshold.

    Returns:
        V: The evaluated state-value array, shape (num_states,).
    """
    num_states = P.shape[0]
    V = np.zeros(num_states)

    # Your vectorized implementation here
    # V[s] = \sum policy(s, a) * (reward(s, a) + gamma * \sum_s' P(s'|s, a) V[s'])
    # .    = \sum_a policy(s, a) * reward(s, a) + gamma * \sum_a' \sum_s' (policy(s, a) * P(s'|s, a) * V[s'])
    P_pi = (P * policy[:, :, None]).sum(1)
    R_pi = (policy * R).sum(1)

    V = R_pi + gamma * P_pi.T @ V

    return V
