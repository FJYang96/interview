# Problem statement:
# You are given a discrete-time linear system x_{t+1} = A x_t + B u_t under double integrator dynamics.
# Implement the finite horizon LQR solution for the horizon N.

import numpy as np


def double_integrator_matrices(dt):
    """
    Discrete-time double integrator with acceleration input.

    State:
        x = [position, velocity]

    Control:
        u = acceleration

    Args:
        dt: timestep.

    Returns:
        A: shape (2, 2)
        B: shape (2, 1)
    """
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt**2], [dt]])
    return A, B


def finite_horizon_lqr(A, B, Q, R, Qf, N):
    """
    Finite-horizon discrete-time LQR.

    Args:
        A: shape (n, n)
        B: shape (n, m)
        Q: shape (n, n)
        R: shape (m, m)
        Qf: shape (n, n)
        N: horizon length

    Returns:
        K_list: shape (N, m, n), feedback gains.
    """
    n, m = B.shape
    K_list = np.zeros((N, m, n))
    P = Qf
    for t in reversed(range(N)):
        K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        K_list[t] = K
    return K_list


def rollout_linear(A, B, K_list, x0, u_min=None, u_max=None):
    """
    Roll out the closed-loop linear system.

    Args:
        A: shape (n, n)
        B: shape (n, m)
        K_list: shape (N, m, n)
        x0: shape (n,)
        u_min: optional scalar or shape (m,)
        u_max: optional scalar or shape (m,)

    Returns:
        xs: shape (N + 1, n)
        us: shape (N, m)
    """
    n, m = B.shape
    N = K_list.shape[0]

    xs = np.zeros((N + 1, n))
    us = np.zeros((N, m))
    xs[0] = x0
    for t in range(N):
        us[t] = -K_list[t] @ xs[t]
        xs[t + 1] = A @ xs[t] + B @ us[t]
    return xs, us


def trajectory_cost(xs, us, Q, R, Qf):
    """
    Compute finite-horizon quadratic cost.
    """
    cost = float(xs[-1] @ Qf @ xs[-1])

    for t in range(us.shape[0]):
        cost += float(xs[t] @ Q @ xs[t] + us[t] @ R @ us[t])

    return cost


def test_double_integrator_matrices():
    dt = 0.1
    A, B = double_integrator_matrices(dt)

    expected_A = np.array(
        [
            [1.0, 0.1],
            [0.0, 1.0],
        ]
    )

    expected_B = np.array(
        [
            [0.005],
            [0.1],
        ]
    )

    np.testing.assert_allclose(A, expected_A, atol=1e-12)
    np.testing.assert_allclose(B, expected_B, atol=1e-12)


def test_lqr_shapes_and_stabilization():
    dt = 0.05
    N = 80

    A, B = double_integrator_matrices(dt)

    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    Qf = np.diag([100.0, 10.0])

    K_list = finite_horizon_lqr(A, B, Q, R, Qf, N)

    assert K_list.shape == (N, 1, 2)

    x0 = np.array([2.0, 0.0])
    xs, us = rollout_linear(A, B, K_list, x0)

    assert xs.shape == (N + 1, 2)
    assert us.shape == (N, 1)

    assert np.linalg.norm(xs[-1]) < 1e-2


def test_lqr_cost_beats_zero_control():
    dt = 0.05
    N = 80

    A, B = double_integrator_matrices(dt)

    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    Qf = np.diag([100.0, 10.0])

    K_list = finite_horizon_lqr(A, B, Q, R, Qf, N)

    x0 = np.array([2.0, 0.0])

    xs_lqr, us_lqr = rollout_linear(A, B, K_list, x0)

    K_zero = np.zeros_like(K_list)
    xs_zero, us_zero = rollout_linear(A, B, K_zero, x0)

    cost_lqr = trajectory_cost(xs_lqr, us_lqr, Q, R, Qf)
    cost_zero = trajectory_cost(xs_zero, us_zero, Q, R, Qf)

    assert cost_lqr < cost_zero


if __name__ == "__main__":
    test_double_integrator_matrices()
    test_lqr_shapes_and_stabilization()
    test_lqr_cost_beats_zero_control()
    print("Exercise 2 tests passed.")
