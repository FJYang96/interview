import numpy as np


def reset(self, seed=None):
    if seed is not None or not hasattr(self, "rng"):
        self.rng = np.random.default_rng(seed)

    q0 = self.q0
    qd0 = self.qd0
    base_pos0 = self.base_pos0
    base_vel0 = np.zeros(3)
    base_ori0 = self.base_ori0
    base_omega0 = np.zeros(3)

    for _ in range(100):
        q = q0 + self.rng.uniform(-0.2, 0.2, size=q0.shape)
        q = np.clip(q, -np.pi, np.pi)

        qd = qd0 + self.rng.uniform(-0.05, 0.05, size=qd0.shape)
        base_pos = base_pos0 + self.rng.uniform(-0.05, 0.05, size=base_pos0.shape)

        if base_pos[2] >= 0.2 and self.feet_not_penetrating(base_pos, q):
            break
    else:
        q = q0
        qd = qd0
        base_pos = base_pos0

    self.q = q
    self.qd = qd
    self.base_pos = base_pos
    self.base_vel = base_vel0
    self.base_ori = base_ori0
    self.base_omega = base_omega0

    self.mass = self.rng.uniform(0.8, 1.2)
    self.friction = self.rng.uniform(0.3, 1.0)

    self.a_prev = np.zeros(6)

    self.t = 0
    self.done = False

    return self._get_obs()
