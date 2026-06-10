# Problem Statement:
#   (1) Given a pendulum with torque input with mass m at the tip and length l. Derive the dynamics equation for the angle \theta.
#   (2) Implement a simulation pipeline using semi-implicit Euler method; Clip the angle to be between [-pi, pi)
#   (3) Design a PD controller for the system

import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pendulum_accel(x, tau, params):
    """
    Compute angular acceleration for a torque-controlled pendulum.

    Args:
        x: shape (2,), [theta, theta_dot]
        tau: scalar torque
        params: dict with keys m, l, b, g

    Returns:
        theta_ddot: scalar
    """
    m, l, b, g = params["m"], params["l"], params["b"], params["g"]
    th, th_dot = x
    return tau - b * th_dot - m * g * l * np.sin(th)


def pendulum_step(x, tau, dt, params):
    """
    One semi-implicit Euler step.

    Args:
        x: shape (2,), [theta, theta_dot]
        tau: scalar
        dt: timestep
        params: dict

    Returns:
        x_next: shape (2,)
    """
    th, th_dot = x
    th_dot_ = th_dot + dt * pendulum_accel(x, tau, params)
    th_ = th + dt * th_dot_
    return np.array([th_, th_dot_])


def pd_pendulum_control(
    x, theta_des, theta_dot_des, kp, kd, tau_max, params, gravity_comp
):
    """
    PD controller for pendulum angle tracking.

    Args:
        x: shape (2,), [theta, theta_dot]
        theta_des: desired angle
        theta_dot_des: desired angular velocity
        kp: proportional gain
        kd: derivative gain
        tau_max: torque limit
        params: dict
        gravity_comp: whether to add gravity/damping compensation

    Returns:
        tau: scalar torque after clipping.
    """
    theta, theta_dot = x
    err = wrap_to_pi(theta_des - theta)
    err_d = theta_dot_des - theta_dot
    tau = kp * err + kd * err_d
    if gravity_comp:
        m, l, b, g = params["m"], params["l"], params["b"], params["g"]
        tau += b * theta_dot + m * g * l * np.sin(theta)

    tau = tau.clip(-tau_max, tau_max)
    return tau


def simulate_pendulum(x0, theta_des, T, dt, params, kp, kd, tau_max, gravity_comp):
    """
    Simulate closed-loop pendulum control.

    Args:
        x0: shape (2,), initial [theta, theta_dot]
        theta_des: desired angle
        T: number of timesteps
        dt: timestep
        params: dict
        kp, kd: controller gains
        tau_max: torque limit
        gravity_comp: bool

    Returns:
        xs: shape (T + 1, 2)
        us: shape (T,)
    """
    xs = np.zeros((T + 1, 2))
    us = np.zeros(T)
    xs[0] = x0
    for t in range(T):
        tau = pd_pendulum_control(
            xs[t], theta_des, 0, kp, kd, tau_max, params, gravity_comp
        )
        us[t] = tau
        xs[t + 1] = pendulum_step(xs[t], tau, dt, params)
    return xs, us


def test_wrap_to_pi():
    assert np.isclose(wrap_to_pi(0.0), 0.0)
    assert np.isclose(wrap_to_pi(2.0 * np.pi), 0.0)
    assert np.isclose(wrap_to_pi(-2.0 * np.pi), 0.0)

    wrapped = wrap_to_pi(3.5 * np.pi)
    assert -np.pi <= wrapped < np.pi


def test_pendulum_accel_known_case():
    params = {
        "m": 1.0,
        "l": 1.0,
        "b": 0.0,
        "g": 9.81,
    }

    x = np.array([np.pi / 2.0, 0.0])
    tau = 0.0

    theta_ddot = pendulum_accel(x, tau, params)

    # At theta = pi/2 with zero torque, acceleration is -g/l.
    assert np.isclose(theta_ddot, -9.81, atol=1e-9)


def test_pendulum_control_reduces_error():
    params = {
        "m": 1.0,
        "l": 1.0,
        "b": 0.1,
        "g": 9.81,
    }

    x0 = np.array([1.0, 0.0])
    theta_des = 0.0

    xs, us = simulate_pendulum(
        x0=x0,
        theta_des=theta_des,
        T=400,
        dt=0.01,
        params=params,
        kp=30.0,
        kd=10.0,
        tau_max=20.0,
        gravity_comp=True,
    )

    initial_error = abs(wrap_to_pi(theta_des - xs[0, 0]))
    final_error = abs(wrap_to_pi(theta_des - xs[-1, 0]))

    assert xs.shape == (401, 2)
    assert us.shape == (400,)

    assert np.max(np.abs(us)) <= 20.0 + 1e-9
    assert final_error < initial_error
    assert final_error < 0.05


if __name__ == "__main__":
    test_wrap_to_pi()
    test_pendulum_accel_known_case()
    test_pendulum_control_reduces_error()
    print("Exercise 3 tests passed.")
