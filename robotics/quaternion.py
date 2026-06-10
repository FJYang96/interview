import numpy as np


def quat_mul(q1, q2):
    """
    Quaternion multiplication in [w, x, y, z] order.
    """
    s1, v1 = q1[0], q1[1:]
    s2, v2 = q2[0], q2[1:]

    q_prod = np.zeros(4)
    q_prod[0] = s1 * s2 - np.dot(v1, v2)
    q_prod[1:] = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return q_prod


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_from_rotvec(phi):
    """
    Exponential map from rotation vector to quaternion.

    phi: shape (3,)
    returns: shape (4,), [w, x, y, z]
    """
    theta = np.linalg.norm(phi)
    dq = np.zeros(4)
    if theta > 1e-10:
        dq[0] = np.cos(theta / 2)
        dq[1:] = np.sin(theta / 2) * phi / theta
    else:
        dq[0] = 1
        dq[1:] = phi / 2
    return quat_normalize(dq)


def integrate_quaternion_body(
    q: np.ndarray, omega_body: np.ndarray, dt: float
) -> np.ndarray:
    """
    Args:
        q: shape (4,), unit quaternion in [w, x, y, z] order.
        omega_body: shape (3,), angular velocity expressed in body frame, rad/s.
        dt: timestep in seconds.

    Returns:
        q_next: shape (4,), unit quaternion after one timestep.
    """
    q = quat_normalize(q)
    dq = quat_from_rotvec(omega_body * dt)
    return quat_normalize(quat_mul(q, dq))


def integrate_quaternion(q, omega, dt, frame="body"):
    q = quat_normalize(q)
    dq = quat_from_rotvec(omega * dt)

    if frame == "body":
        q_next = quat_mul(q, dq)
    elif frame == "world":
        q_next = quat_mul(dq, q)
    else:
        raise ValueError("frame must be either 'body' or 'world'")

    return quat_normalize(q_next)


############################ TESTS ####################################


def assert_quat_same_rotation(q1, q2, tol=1e-9):
    """
    q and -q represent the same rotation, so compare up to sign.
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    err1 = np.linalg.norm(q1 - q2)
    err2 = np.linalg.norm(q1 + q2)

    assert min(err1, err2) < tol, (q1, q2, min(err1, err2))


def random_unit_quat(rng):
    q = rng.normal(size=4)
    return quat_normalize(q)


def test_identity_zero_velocity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    dt = 0.01

    q_next = integrate_quaternion_body(q, omega, dt)
    assert_quat_same_rotation(q_next, q)


def test_known_z_rotation():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, np.pi / 2])
    dt = 1.0

    q_next = integrate_quaternion_body(q0, omega, dt)

    expected = np.array(
        [
            np.cos(np.pi / 4),
            0.0,
            0.0,
            np.sin(np.pi / 4),
        ]
    )

    assert_quat_same_rotation(q_next, expected)


def test_unit_norm_randomized():
    rng = np.random.default_rng(0)

    for _ in range(1000):
        q = random_unit_quat(rng)
        omega = rng.normal(size=3)
        dt = rng.uniform(1e-4, 0.1)

        q_next = integrate_quaternion_body(q, omega, dt)

        assert abs(np.linalg.norm(q_next) - 1.0) < 1e-12


def test_body_update_matches_manual_right_multiplication():
    rng = np.random.default_rng(1)

    for _ in range(1000):
        q = random_unit_quat(rng)
        omega = rng.normal(size=3)
        dt = rng.uniform(1e-4, 0.1)

        dq = quat_from_rotvec(omega * dt)
        expected = quat_normalize(quat_mul(q, dq))

        actual = integrate_quaternion(q, omega, dt, frame="body")

        assert_quat_same_rotation(actual, expected)


def test_world_update_matches_manual_left_multiplication():
    rng = np.random.default_rng(2)

    for _ in range(1000):
        q = random_unit_quat(rng)
        omega = rng.normal(size=3)
        dt = rng.uniform(1e-4, 0.1)

        dq = quat_from_rotvec(omega * dt)
        expected = quat_normalize(quat_mul(dq, q))

        actual = integrate_quaternion(q, omega, dt, frame="world")

        assert_quat_same_rotation(actual, expected)


def test_body_and_world_differ_generically():
    rng = np.random.default_rng(3)

    found_difference = False

    for _ in range(100):
        q = random_unit_quat(rng)
        omega = rng.normal(size=3)
        dt = 0.05

        q_body = integrate_quaternion(q, omega, dt, frame="body")
        q_world = integrate_quaternion(q, omega, dt, frame="world")

        if (
            min(np.linalg.norm(q_body - q_world), np.linalg.norm(q_body + q_world))
            > 1e-6
        ):
            found_difference = True
            break

    assert found_difference


def test_small_angle_stability():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([1e-10, -2e-10, 3e-10])
    dt = 1e-3

    q_next = integrate_quaternion_body(q, omega, dt)

    assert np.all(np.isfinite(q_next))
    assert abs(np.linalg.norm(q_next) - 1.0) < 1e-12


def run_all_tests():
    test_identity_zero_velocity()
    test_known_z_rotation()
    test_unit_norm_randomized()
    test_body_update_matches_manual_right_multiplication()
    test_world_update_matches_manual_left_multiplication()
    test_body_and_world_differ_generically()
    test_small_angle_stability()

    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
