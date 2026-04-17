import matplotlib.pyplot as plt
import numpy as np

# hyperparameters
threshold = 1e-3
gamma = 0.99

# states
n, m = 4, 4  # grid size
S = n * m


def state_ind_2_coord(ind):
    return (ind // n, ind % n)


# actions
A = 4
action_2_dir = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


# transition
def transition(state, action):
    i, j = state_ind_2_coord(state)
    di, dj = action_2_dir[action]
    return max(0, min(n - 1, i + di)) * n + max(0, min(m - 1, j + dj))


# reward
def reward(state, action):
    return -1 if (state != 0 or state != (n * m - 1)) else 0


# policy to evaluation
def uniform_policy(action, state):
    return 1 / A


# Policy evaluation loop
V = [0] * (n * m)
while True:
    delta_value = 0
    for s in range(S):
        if s == 0 or s == (n * m - 1):
            continue
        # V[s] = \sum_a [ \pi(a|s) (r(s, a) + gamma * \sum_s' P(s'|s, a) V[s'])
        new_value = 0
        for a in range(A):
            new_value += uniform_policy(a, s) * (
                reward(s, a) + gamma * V[transition(s, a)]
            )
        delta_value = max(delta_value, abs(new_value - V[s]))
        V[s] = new_value

    if delta_value < threshold:
        break

V = np.array(V).reshape(n, m)
print(V)
plt.imshow(V)
plt.show()
