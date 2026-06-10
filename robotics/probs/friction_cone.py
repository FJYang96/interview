# Problem Statement:
# A simplified humanoid torso is supported by two point contacts (feet). Given the feet positions relative to the Center of Mass (CoM) and
# a desired 6-DoF stabilization wrench at the CoM, compute the 3D forces for each foot. Formulate this as a Quadratic Program to minimize
# the wrench tracking error, subject to unilateral contact and linearized friction cone constraints.

import cvxpy as cp
import numpy as np


def skew(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


def distribute_wrench(V_des, p1, p2, mu):
    """
    Solves for contact forces f1, f2 to achieve V_des.
    V_des: (6,) desired wrench [f_x, f_y, f_z, tau_x, tau_y, tau_z]
    p1, p2: (3,) positions of feet relative to CoM
    mu: float, friction coefficient
    Returns: f: (6,) stacked [f1_x, f1_y, f1_z, f2_x, f2_y, f2_z]
    """
    # Define the decision variable
    f = cp.Variable(6)

    # Construct the wrench V = W f
    W = np.block([[np.eye(3), np.eye(3)], [skew(p1), skew(p2)]])

    # QP cost: 0.5 * ||V - W f||^2 => 0.5 * f^\top (W^\top W) f + V_des^\top W f
    alpha = 1e-4
    objective = cp.Minimize(
        cp.quad_form(f, W.T @ W + np.eye(6) * alpha) - 2 * f @ (W.T @ V_des)
    )

    # QP Constraint
    constr = [
        -f[2] <= 0,  # Foot 1 force upward
        -f[5] <= 0,  # Foot 2 force upward
        cp.norm1(f[:2]) <= f[2],  # Friction cone
        cp.norm1(f[3:5]) <= f[5],  # Friction cone
    ]

    # Solve and return f
    prob = cp.Problem(objective, constr)
    prob.solve()
    return f.value


def test_distribute_wrench():
    V_des = np.array([0, 0, 98.1, 0, 0, 0])  # 10kg robot resisting gravity
    p1 = np.array([0, 0.2, -1.0])
    p2 = np.array([0, -0.2, -1.0])
    mu = 0.5

    f_opt = distribute_wrench(V_des, p1, p2, mu)

    f1 = f_opt[0:3]
    f2 = f_opt[3:6]

    # Test Unilateral
    assert f1[2] >= -1e-5 and f2[2] >= -1e-5

    # Test Friction Cone
    assert abs(f1[0]) <= mu * f1[2] + 1e-5
    assert abs(f1[1]) <= mu * f1[2] + 1e-5

    # Test Wrench tracking (should be near perfect for this feasible target)
    I = np.eye(3)
    W = np.block([[I, I], [skew(p1), skew(p2)]])
    V_achieved = W @ f_opt
    np.testing.assert_allclose(V_achieved, V_des, atol=1e-2)


test_distribute_wrench()
