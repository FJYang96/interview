from typing import Tuple

import numpy as np
from scipy.linalg import expm


def skew_6(V: np.ndarray) -> np.ndarray:
    w, v = V[:3], V[3:]
    return np.array(
        [
            [0, -w[2], w[1], v[0]],
            [w[2], 0, -w[0], v[1]],
            [-w[1], w[0], 0, v[2]],
            [0, 0, 0, 0],
        ]
    )


def adjoint(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    p_skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = p_skew @ R
    return Ad


def compute_floating_base_jacobian(
    T_sb: np.ndarray, M: np.ndarray, Slist: np.ndarray, q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    n = len(q)
    T_arm = np.eye(4)
    J_arm = np.zeros((6, n))

    # Compute Space Jacobian of the arm (relative to base frame)
    for i in range(n):
        J_arm[:, i] = adjoint(T_arm) @ Slist[:, i]
        T_arm = T_arm @ expm(skew_6(Slist[:, i]) * q[i])

    T_se = T_sb @ T_arm @ M

    # J_fb = [Ad_Tsb | Ad_Tsb * J_arm]
    Ad_Tsb = adjoint(T_sb)
    J_fb = np.hstack((Ad_Tsb, Ad_Tsb @ J_arm))

    return T_se, J_fb


# --- Tests ---
def test_floating_base_jacobian():
    T_sb = np.eye(4)
    T_sb[:3, 3] = [1.0, 0.0, 0.0]  # Base shifted by 1 in X

    M = np.eye(4)
    M[:3, 3] = [0.0, 0.0, 2.0]  # EE 2 units along Z from base

    # 2-DoF planar arm along Y-axis
    Slist = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # w_x
            [1, 1, 0, 0, 0, 0],  # w_y (Revolute joints about Y)
            [0, 0, 0, 0, 0, 0],  # w_z
            [0, 0, 1, 1, 0, 0],  # v_x
            [0, 0, 0, 0, 1, 1],  # v_y
            [0, 0, 0, 0, 0, 0],  # v_z
        ]
    ).T
    Slist = Slist[:, :2]  # taking first 2 columns for 2-DoF
    q = np.array([np.pi / 2, 0.0])

    T_se, J_fb = compute_floating_base_jacobian(T_sb, M, Slist, q)

    assert T_se.shape == (4, 4)
    assert J_fb.shape == (6, 8)  # 6 DoF base + 2 DoF arm
    print("Problem 1 Tests Passed.")


test_floating_base_jacobian()
