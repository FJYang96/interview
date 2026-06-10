# Problem statement:
#   You are given an n-link planar robot arm. Each joint is revolute. Given joint angles q, link lengths lengths,
#   and a Cartesian target position target, implement one step of a Jacobian-based controller that moves the end-effector toward the target.
# The controller should use:
#   v_des = K_p (x_target - x),
# followed by damped least squares
#   \dot q = (J^\top J + \lambda I)^{-1} J^\top v_des
# Then, clip joint velocities and integrate:
#   q_next = q + dt \dot q

import numpy as np


def fk_planar(q, lengths):
    """
    Forward kinematics for an n-link planar revolute arm.

    Args:
        q: shape (n,), joint angles in radians.
        lengths: shape (n,), link lengths.

    Returns:
        ee_pos: shape (2,), end-effector position [x, y].
    """
    lengths = np.asarray(lengths)
    q_cum = np.cumsum(q)
    dx = np.cos(q_cum) * lengths
    dy = np.sin(q_cum) * lengths
    return np.array([dx.sum(), dy.sum()])


def jacobian_planar(q, lengths):
    """
    Geometric Jacobian mapping joint velocities to end-effector velocity.

    Args:
        q: shape (n,)
        lengths: shape (n,)

    Returns:
        J: shape (2, n), where xdot = J @ qdot.
    """
    lengths = np.asarray(lengths, dtype=float)
    q_cum = np.cumsum(q)
    x_pos = np.concatenate(([0], np.cumsum(np.cos(q_cum) * lengths)))
    y_pos = np.concatenate(([0], np.cumsum(np.sin(q_cum) * lengths)))
    J = np.vstack(
        [
            -(y_pos[-1] - y_pos[:-1]),
            x_pos[-1] - x_pos[:-1],
        ]
    )
    return J


def dls_ik_step(q, lengths, target, kp=2.0, dt=0.05, lam=1e-2, dq_max=5.0):
    """
    One damped least-squares IK control step.

    Args:
        q: shape (n,), current joint angles.
        lengths: shape (n,), link lengths.
        target: shape (2,), desired end-effector position.
        kp: Cartesian proportional gain.
        dt: integration timestep.
        lam: damping factor.
        dq_max: scalar joint velocity limit.

    Returns:
        q_next: shape (n,)
        dq: shape (n,), commanded joint velocity after clipping.
    """
    q = np.asarray(q, dtype=float)
    target = np.asarray(target, dtype=float)

    x = fk_planar(q, lengths)
    J = jacobian_planar(q, lengths)

    v_des = kp * (target - x)

    n = q.size
    A = J.T @ J + (lam**2) * np.eye(n)
    b = J.T @ v_des

    dq = np.linalg.solve(A, b)
    dq = np.clip(dq, -dq_max, dq_max)

    q_next = q + dt * dq

    return q_next, dq


def test_fk_planar_known_cases():
    lengths = np.array([1.0, 1.0])

    q = np.array([0.0, 0.0])
    np.testing.assert_allclose(fk_planar(q, lengths), np.array([2.0, 0.0]), atol=1e-9)

    q = np.array([np.pi / 2, 0.0])
    np.testing.assert_allclose(fk_planar(q, lengths), np.array([0.0, 2.0]), atol=1e-9)

    q = np.array([0.0, np.pi / 2])
    np.testing.assert_allclose(fk_planar(q, lengths), np.array([1.0, 1.0]), atol=1e-9)


def test_jacobian_planar_against_finite_difference():
    q = np.array([0.4, -0.7, 0.2])
    lengths = np.array([1.0, 0.8, 0.6])

    J = jacobian_planar(q, lengths)

    eps = 1e-6
    J_fd = np.zeros_like(J)

    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = eps

        x_plus = fk_planar(q + dq, lengths)
        x_minus = fk_planar(q - dq, lengths)

        J_fd[:, i] = (x_plus - x_minus) / (2.0 * eps)

    np.testing.assert_allclose(J, J_fd, atol=1e-6)


def test_dls_ik_step_reduces_distance_to_target():
    q = np.array([0.4, -0.7, 0.2])
    lengths = np.array([1.0, 0.8, 0.6])
    target = np.array([1.6, 0.7])

    dist_before = np.linalg.norm(target - fk_planar(q, lengths))

    q_next, dq = dls_ik_step(
        q=q,
        lengths=lengths,
        target=target,
        kp=2.0,
        dt=0.05,
        lam=1e-2,
        dq_max=10.0,
    )

    dist_after = np.linalg.norm(target - fk_planar(q_next, lengths))

    assert q_next.shape == q.shape
    assert dq.shape == q.shape
    assert dist_after < dist_before


if __name__ == "__main__":
    test_fk_planar_known_cases()
    test_jacobian_planar_against_finite_difference()
    test_dls_ik_step_reduces_distance_to_target()
    print("Exercise 1 tests passed.")
