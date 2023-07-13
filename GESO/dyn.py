import fym
import numpy as np
from fym.utils.rot import quat2dcm

from ftc.utils import safeupdate


class Quadrotor(fym.BaseEnv):
    m = 1.79
    g = 9.81
    l = 0.18 # [m]
    kT = 8.82e-6 # [kg*m]
    kD = 1.09e-7 # [kg*m^2]
    B = np.array(
        [
            [0, -l*kT, 0, l*kT],
            [l*kT, 0, -l*kT, 0],
            [-kD, kD, -kD, kD],
        ]
    )
    J = np.diag([0.01335, 0.01335, 0.02465])
    
    ENV_CONFIG = {
        "init": {
            "quat": np.vstack((1, 0, 0, 0)),
            "omega": np.zeros((3, 1)),
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__()
        self.quat = fym.BaseSystem(env_config["init"]["quat"])
        self.omega = fym.BaseSystem(env_config["init"]["omega"])


    def deriv(self, quat, omega, u, dtrb):
        J = self.J
        Jinv = np.linalg.inv(J)
        p, q, r = np.ravel(omega)
        dquat = 0.5 * np.array([[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]]).dot(quat)
        domega = Jinv @ (
            - np.cross(omega, J.dot(omega), axis=0)
            + u
            + dtrb
        )

        return dquat, domega

    def set_dot(self, t, u, dtrb):
        quat, omega = self.observe_list()
        dots = self.deriv(quat, omega, u, dtrb)
        self.quat.dot, self.omega.dot = dots

    def disturbance(self, t):
        num = 7
        w = (np.random.rand(num, 1) * 2.45 + 0.05) * np.pi
        p = np.random.rand(num, 1) * 10
        a = np.random.rand(num, 1) * 2
        d0 = 1
        
        dtrb_wind = d0
        for i in range(num):
            dtrb_wind = dtrb_wind + a[i] * np.sin(w[i] * t + p[i])

        dtrb_info = {
            "freq": w,
            "phase shift": p,
            "amplitude of sinusoid": a,
        }
        return dtrb_wind, dtrb_info

if __name__ == "__main__":
    system = Quadrotor()
    print(system.disturbance(10))


