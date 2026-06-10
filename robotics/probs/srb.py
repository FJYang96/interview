# Problem Statement:
# Given the continuous time dynamics matrices $A$ and $B$ for a linearized Single Rigid Body Model,
# perform a zero-order hold Forward Euler discretization using time step $dt$. Then, for a prediction
# horizon $N$, construct the dense matrices $\mathbf{S}_x$ and $\mathbf{S}_u$ such that the stacked state
# vector across the entire horizon $\mathbf{X}$ can be evaluated directly via
#       $\mathbf{X} = \mathbf{S}_x x_0 + \mathbf{S}_u \mathbf{U}$.

import numpy as np


def build_mpc_matrices(A, B, dt, N):
    """
    A: (n, n) continuous state matrix
    B: (n, m) continuous input matrix
    dt: float, time step
    N: int, prediction horizon
    Returns: Sx: (n*N, n), Su: (n*N, m*N)
    """
    # Construct the discrete time (A_d, B_d)
    n, m = B.shape
    Ad = np.eye(n) + A * dt
    Bd = B * dt

    # Construct the S matrices
    Sx = np.zeros((N, n, n))
    Su = np.zeros((N, n, N, m))
    Ad_k = np.eye(n)
    for i in range(N):
        Ad_k = Ad_k @ Ad
        Sx[i] = Ad_k
        for j in range(i + 1):
            if i == j:
                Su[i, :, j, :] = Bd
            else:
                Su[i, :, j, :] = Sx[i - j - 1] @ Bd
    Sx = Sx.reshape(-1, n)
    Su = Su.reshape(N * n, N * m)
    return Sx, Su


def test_build_mpc_matrices():
    # Simple double integrator: x_dot = v, v_dot = u
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    dt = 0.1
    N = 3

    Sx, Su = build_mpc_matrices(A, B, dt, N)

    assert Sx.shape == (6, 2)
    assert Su.shape == (6, 3)

    x0 = np.random.randn(2)
    U = np.random.randn(3)

    # Predict manually
    Ad = np.eye(2) + A * dt
    Bd = B * dt

    x1 = Ad @ x0 + Bd @ U[0:1]
    x2 = Ad @ x1 + Bd @ U[1:2]
    x3 = Ad @ x2 + Bd @ U[2:3]

    X_manual = np.concatenate([x1, x2, x3])

    # Predict using matrices
    X_dense = Sx @ x0 + Su @ U

    np.testing.assert_allclose(X_dense, X_manual, atol=1e-10)
    print("Passed!")


test_build_mpc_matrices()
