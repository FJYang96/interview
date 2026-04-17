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
gamma = 0.9

V = [0] * 16

while True:
    delta = 0
    V_new = [0] * 16
    for s in range(S):
        v_new = 0
        for a in range(A):
            v_new += 0.25 * (reward(s, a) + gamma * V[transition(s, a)])
        V_new[s] = v_new
        delta = max(delta, abs(V[s] - V_new[s]))
    if delta < tol:
        break
    for s in range(S):
        V[s] = V_new[s]
    print(f"delta={delta}")

V = np.array(V).reshape(4, 4)
print(V)
plt.figure()
plt.imshow(V)
plt.show()
