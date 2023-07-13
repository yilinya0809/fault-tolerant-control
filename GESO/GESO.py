import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm


class GESOController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.J = env.plant.J


    def get_control(self, t, env):
        quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        quatd, omegad = env.get_ref(t, "quatd", "omegad")
        angd = np.vstack(quat2angle(quatd)[::-1])
        omegaddot = 0
        quat_0e = quat[0] * quatd[0] + quat[1:].T * quatd[1:]
        quat_ve = quatd[0] * quat[1:] - quat[0] * quatd[1:] + np.cross(quat[1:], quatd[1:], axis=0)
        R = quat2dcm(quat)
        omega_e = omega - R.dot(omegad)
        quat_ve_dot = 0.5 * np.cross(quat_ve, omega_e, axis=0) + quat_0e * omega_e

        K1 = np.diag([1, 1, 1])
        K2 = np.diag([1, 1, 1])

        omega_ed = - K1.dot(quat_ve)
        omega_e_til = omega_e - omega_ed
        N = self.J.dot(np.cross(omega_e, R.dot(omegad), axis=0) - R.dot(omegaddot)) - np.cross(omega, self.J.dot(omega), axis=0) + self.J * K1 * quat_ve_dot + quat_ve

        u_nominal = -N - K2*omega_e_til
        controller_info = {
            "quatd": quatd,
            "omegad": omegad,
            "angd": angd,
            "ang": ang,
        }

        return u_nominal, controller_info

    def geso(self, t, env):
        
        quat, omega = env.plant.observe_list()

        # x = np.vstack((J.dot(omega), dtrb))
        A = np.concatenate((np.concatenate((np.zeros((3,3)), np.eye(3)), axis=1), np.zeros((3, 6))), axis=0)
        B = np.concatenate((np.eye(3), np.zeros((3,3))), axis=0)
        Bd = np.concatenate((np.zeros((3,3)), np.eye(3)), axis=0)
        C = np.concatenate((np.eye(3), np.zeros((3,3))), axis=1)
        l0 = 40
        l1 = 400
        L = np.concatenate((l0 * np.eye(3), l1 * np.eye(3)), axis=0)
        

        dhat = 0
        u_dtrb = -dhat

        observer_info = {}
        
        
        
        return u_dtrb, observer_info


