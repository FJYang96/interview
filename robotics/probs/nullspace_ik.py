# Problem Statement:
# A 3-link planar robotic leg requires continuous tracking. Formulate a differential Inverse Kinematics ($\dot{q}$) solver.
# The primary (strict) task is to keep the foot flat (pitch angular velocity $\dot{\theta}_{foot} = 0$). The secondary task
# is to track a target spatial velocity for the foot's position, $\dot{x}_{target} \in \mathbb{R}^2$, projected into the null
# space of the primary task.

import numpy as np


def solve_prioritized_ik(q, L, v_pri, v_target):
    """
    q: (3,) joint angles
    L: (3,) link lengths [l1, l2, l3]
    v_pri: (1,) foot angular velocity
    v_target: (2,) target spatial velocity [vx, vy]
    Returns: dq: (3,) joint velocities
    """
    # Primary task Jacobian. d omega_3 / d q_dot = [1, 1, 1]
    J1 = np.ones((1, 3))

    # Secondary task Jacobian: end effector Jacobian
    q_cum = np.cumsum(q)
    dx = L * np.cos(q_cum)
    dy = L * np.sin(q_cum)
    x_pos = np.concatenate([[0], np.cumsum(dx)])
    y_pos = np.concatenate([[0], np.cumsum(dy)])
    J2 = np.vstack([-(y_pos[-1] - y_pos[:-1]), x_pos[-1] - x_pos[:-1]])

    # Find the correct dq
    J1_inv = np.linalg.pinv(J1)
    dq_pri = J1_inv @ v_pri
    N_pri = np.eye(3) - J1_inv @ J1
    dq_sec = np.linalg.pinv(J2 @ N_pri) @ (v_target - J2 @ J1_inv @ v_pri)
    dq = dq_pri + N_pri @ dq_sec

    return dq


def test_solve_prioritized_ik():
    q = np.array([np.pi / 4, -np.pi / 2, np.pi / 4])
    L = np.array([0.5, 0.5, 0.2])
    v_target = np.array([0.1, -0.05])
    v_pri = np.array([1.0])

    dq = solve_prioritized_ik(q, L, v_pri, v_target)

    # Test Primary constraint: sum of joint velocities must be exactly 0
    assert np.isclose(np.sum(dq), v_pri, atol=1e-10)

    # Test Secondary constraint mapping (best effort given projection)
    # Calculate actual achieved end-effector velocity
    q1, q2, q3 = q
    l1, l2, l3 = L
    c1, s1 = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    c123, s123 = np.cos(q1 + q2 + q3), np.sin(q1 + q2 + q3)
    J_sec = np.array(
        [
            [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
            [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123],
        ]
    )

    v_achieved = J_sec @ dq
    # It should track perfectly here because the arm is not in a singular
    # configuration and 2 task DoF + 1 pri DoF = 3 joint DoFs.
    np.testing.assert_allclose(v_achieved, v_target, atol=1e-10)
    print("Passed!")


test_solve_prioritized_ik()
