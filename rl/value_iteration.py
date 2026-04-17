import matplotlib.pyplot as plt
import numpy as np

S = 16
A = 4

# action is 0-3 from top down left right
delta_action = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def transition(state, action):
    if state == 0 or state == 15:
        return state
    x, y = state // 4, state % 4
    dx, dy = delta_action[action]
    x_ = min(max(x + dx, 0), 3)
    y_ = min(max(y + dy, 0), 3)
    return x_ * 4 + y_


def reward(state, action):
    return 0 if (state == 0 or state == 15) else -1


tol = 1e-3
gamma = 0.99

# value iteration
vi_counter = 0
V_star = [0] * 16
while True:
    vi_counter += 1
    delta = 0
    V_new = [0] * 16
    for s in range(S):
        if s == 0 or s == 15:
            continue
        max_val_s = -float("inf")
        for a in range(A):
            v_new = reward(s, a) + gamma * V_star[transition(s, a)]
            max_val_s = max(max_val_s, v_new)
        V_new[s] = max_val_s
        delta = max(delta, abs(V_new[s] - V_star[s]))

    if delta < tol:
        break
    V_star = V_new


def policy_improvement(V):
    pi_ = [-1] * 16
    for s in range(S):
        v_by_a = [reward(s, a) + gamma * V[transition(s, a)] for a in range(A)]
        pi_[s] = np.argmax(v_by_a)
    return pi_


pi_star = policy_improvement(V_star)
V_star = np.array(V_star).reshape(4, 4)
pi_star = np.array(pi_star).reshape(4, 4)


# Policy iteration
def policy_evaluation(pi):
    # pi: state -> action
    counter = 0
    V = [0] * 16
    while True:
        counter += 1
        delta = 0
        V_new = [0] * 16
        for s in range(S):
            if s == 0 or s == 15:
                continue
            a = pi[s]
            V_new[s] = reward(s, a) + gamma * V[transition(s, a)]
            delta = max(delta, abs(V_new[s] - V[s]))

        if delta < tol:
            break
        V = V_new
    return V, counter


pi_PI = [np.random.randint(A) for _ in range(S)]
pi_counter, pi_total_iter = 0, 0
while True:
    pi_counter += 1
    # 1. policy evaluation
    V, pe_ctr = policy_evaluation(pi_PI)
    pi_total_iter += pe_ctr

    # 2. policy improvement
    pi_ = policy_improvement(V)
    if pi_PI == pi_:
        break
    pi_PI = pi_

pi_PI = np.array(pi_PI).reshape(4, 4)
print(f"VI converged in {vi_counter} iterations")
print(pi_star)
print(
    f"PI converged in {pi_counter} outer iterations and a total of {pi_total_iter} iterations"
)
print(pi_PI)
