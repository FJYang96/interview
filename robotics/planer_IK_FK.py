from math import sin

import numpy as np


class PlanarManipulator:
    def __init__(self, link_lengths):
        """
        link_lengths: 1D numpy array of length N
        """
        self.link_lengths = np.array(link_lengths)
        self.n_links = len(link_lengths)

    def forward_kinematics(self, q):
        """
        Computes the (x, y) position of all joint frames and the end-effector.
        q: 1D numpy array of joint angles of length N
        Returns: (N+1, 2) array of (x, y) coordinates. Index 0 is the base (0,0).
        """
        q_cumsum = np.cumsum(q)
        dx = self.link_lengths * np.cos(q_cumsum)
        dy = self.link_lengths * np.sin(q_cumsum)
        coords = np.zeros((self.n_links + 1, 2))
        coords[1:, 0] = np.cumsum(dx)
        coords[1:, 1] = np.cumsum(dy)
        return coords

    def jacobian(self, q):
        """
        Computes the analytical Jacobian matrix for the end-effector position.
        q: 1D numpy array of joint angles
        Returns: (2, N) Jacobian matrix
        """
        fk = self.forward_kinematics(q)  # (N+1, 2)
        jac = np.zeros((2, self.n_links))
        jac[0] = -(fk[-1, 1] - fk[0:-1, 1])
        jac[1] = fk[-1, 0] - fk[0:-1, 0]
        return jac

    def inverse_kinematics(self, target_pos, q_init, max_iters=100, tol=1e-4, lr=0.1):
        """
        Solves for joint angles to reach target_pos using Jacobian pseudo-inverse.
        target_pos: (2,) numpy array (x, y)
        q_init: (N,) numpy array of initial joint angles
        Returns: (N,) numpy array of final joint angles
        """
        q_curr = q_init
        for _ in range(max_iters):
            ee_pos = self.forward_kinematics(q_curr)[-1]
            jac = self.jacobian(q_curr)
            # delta = np.linalg.pinv(jac) @ (target_pos - ee_pos)
            delta = jac.T @ np.linalg.solve(jac @ jac.T, target_pos - ee_pos)
            if np.linalg.norm(delta) < tol:
                break
            q_curr += lr * delta
        return q_curr


class PlanarManipulatorRef:
    def __init__(self, link_lengths):
        """Initializes the N-link planar manipulator."""
        self.link_lengths = np.array(link_lengths, dtype=float)
        self.n_links = len(link_lengths)

    def forward_kinematics(self, q):
        """
        Computes the (x, y) position of all joint frames and the end-effector.
        Vectorized to avoid loops.
        """
        # Cumulative sum calculates the absolute angle of each link
        theta_cum = np.cumsum(q)

        # Calculate x and y displacements for each link
        dx = self.link_lengths * np.cos(theta_cum)
        dy = self.link_lengths * np.sin(theta_cum)

        # Cumulative sum of displacements gives absolute positions.
        # Prepend (0,0) for the base coordinate.
        x = np.concatenate(([0.0], np.cumsum(dx)))
        y = np.concatenate(([0.0], np.cumsum(dy)))

        return np.column_stack((x, y))

    def jacobian(self, q):
        """
        Computes the analytical Jacobian matrix for the end-effector.
        Leverages the 2D cross-product simplification:
        J_i = Z-axis x (EE_pos - Joint_pos) = [-(y_ee - y_i), (x_ee - x_i)]^T
        """
        positions = self.forward_kinematics(q)
        joint_positions = positions[:-1]  # Exclude end-effector
        ee_pos = positions[-1]  # End-effector position

        J = np.zeros((2, self.n_links))
        J[0, :] = -(ee_pos[1] - joint_positions[:, 1])
        J[1, :] = ee_pos[0] - joint_positions[:, 0]

        return J

    def inverse_kinematics(self, target_pos, q_init, max_iters=500, tol=1e-4, lr=0.5):
        """
        Solves for joint angles to reach target_pos using Jacobian pseudo-inverse.
        """
        q = np.array(q_init, dtype=float)
        target_pos = np.array(target_pos, dtype=float)

        for _ in range(max_iters):
            ee_pos = self.forward_kinematics(q)[-1]
            error = target_pos - ee_pos

            if np.linalg.norm(error) < tol:
                break

            J = self.jacobian(q)
            # Use pseudo-inverse to handle singularities robustly
            dq = np.linalg.pinv(J) @ error
            q += lr * dq

        return q


def test_manipulator():
    # Setup a 3-link arm with 1.0 unit links
    lengths = [1.0, 1.0, 1.0]
    arm = PlanarManipulator(lengths)
    # arm_ref = PlanarManipulatorRef(lengths)

    print("Running functional parity tests...")

    # --- Test 1: Forward Kinematics Parity ---
    # q = [0, 0, 0] -> EE should be straight out at (3, 0)
    pos_straight = arm.forward_kinematics([0, 0, 0])
    np.testing.assert_allclose(
        pos_straight[-1], [3.0, 0.0], err_msg="FK failed on zero-angle configuration"
    )

    # q = [pi/2, 0, 0] -> EE should be pointing straight up at (0, 3)
    pos_up = arm.forward_kinematics([np.pi / 2, 0, 0])
    np.testing.assert_allclose(
        pos_up[-1], [0.0, 3.0], atol=1e-15, err_msg="FK failed on rotated configuration"
    )

    # q = [pi/2, -pi/2, -pi/2] -> Should form a zig-zag ending at (1, 1)
    pos_zigzag = arm.forward_kinematics([np.pi / 2, -np.pi / 2, -np.pi / 2])
    np.testing.assert_allclose(
        pos_zigzag[-1],
        [1.0, 0.0],
        atol=1e-15,
        err_msg="FK failed on complex configuration",
    )
    print("✓ Forward Kinematics: Value parity verified.")

    # --- Test 2: Analytical vs Numerical Jacobian Parity ---
    # We use finite differences as the ground truth reference to verify the analytical derivation
    q_test = np.array([0.1, 0.5, -0.3])
    J_analytical = arm.jacobian(q_test)

    epsilon = 1e-6
    J_numerical = np.zeros((2, 3))

    for i in range(3):
        q_plus = q_test.copy()
        q_plus[i] += epsilon
        ee_plus = arm.forward_kinematics(q_plus)[-1]

        q_minus = q_test.copy()
        q_minus[i] -= epsilon
        ee_minus = arm.forward_kinematics(q_minus)[-1]

        # Central difference approximation
        J_numerical[:, i] = (ee_plus - ee_minus) / (2 * epsilon)

    np.testing.assert_allclose(
        J_analytical,
        J_numerical,
        rtol=1e-4,
        err_msg="Jacobian analytical vs numerical mismatch",
    )
    print("✓ Jacobian: Finite-difference numerical parity verified.")

    # --- Test 3: Inverse Kinematics Convergence Parity ---
    target = np.array([1.5, 1.5])
    q_init = np.array([0.1, 0.1, 0.1])

    q_solved = arm.inverse_kinematics(target, q_init, tol=1e-4)
    ee_solved = arm.forward_kinematics(q_solved)[-1]

    # Ensure the final position actually matches the target coordinates
    error = np.linalg.norm(target - ee_solved)
    assert error < 1e-3, f"IK failed to converge to target. Error: {error}"
    print("✓ Inverse Kinematics: Target convergence verified.")
    print("\nAll tests passed successfully.")


def test_against_reference(arm, arm_ref, num_trials=1000):
    """
    Compares the current implementation against a known reference
    using randomized configurations.
    """
    print(f"Running {num_trials} randomized comparative tests...")

    for i in range(num_trials):
        # Generate random joint configurations using randn
        q_rand = np.random.randn(arm.n_links)

        # --- 1. Forward Kinematics Parity ---
        fk_out = arm.forward_kinematics(q_rand)
        fk_ref = arm_ref.forward_kinematics(q_rand)
        np.testing.assert_allclose(
            fk_out,
            fk_ref,
            atol=1e-10,
            err_msg=f"FK mismatch on trial {i} with q={q_rand}",
        )

        # --- 2. Jacobian Parity ---
        jac_out = arm.jacobian(q_rand)
        jac_ref = arm_ref.jacobian(q_rand)
        np.testing.assert_allclose(
            jac_out,
            jac_ref,
            atol=1e-10,
            err_msg=f"Jacobian mismatch on trial {i} with q={q_rand}",
        )

        # --- 3. Inverse Kinematics Parity ---
        # Generate a random reachable target within the workspace
        max_reach = np.sum(arm.link_lengths)
        target_angle = np.random.uniform(-np.pi, np.pi)
        target_radius = np.random.uniform(0.1, max_reach * 0.95)
        target_pos = np.array(
            [target_radius * np.cos(target_angle), target_radius * np.sin(target_angle)]
        )

        q_init = np.random.randn(arm.n_links)

        q_ik_out = arm.inverse_kinematics(target_pos, q_init)
        q_ik_ref = arm_ref.inverse_kinematics(target_pos, q_init)

        # Verify functional parity of the final achieved positions
        ee_out = arm.forward_kinematics(q_ik_out)[-1]
        ee_ref = arm_ref.forward_kinematics(q_ik_ref)[-1]

        np.testing.assert_allclose(
            ee_out,
            ee_ref,
            atol=1e-3,
            err_msg=f"IK end-effector position mismatch on trial {i}",
        )

    print(f"✓ {num_trials} randomized reference comparisons passed successfully.")


# Example integration:
if __name__ == "__main__":
    test_manipulator()

    lengths = [1.0, 1.0, 1.0]
    arm = PlanarManipulator(lengths)  # Your implementation
    arm_ref = PlanarManipulatorRef(lengths)  # The reference implementation

    test_against_reference(arm, arm_ref)
