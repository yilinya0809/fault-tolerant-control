import fym
import numpy as np


class myEnv(fym.BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=5)
        self.pos = fym.BaseSystem(shape=(3, 1))
        self.vel = fym.BaseSystem(np.vstack((0,0,1)))

    def step(self):
        _, done=self.update()
        return done

    def set_dot(self,t):
        pos,vel=self.observe_list()
        self.pos.dot=vel
        self.vel.dot=-2*vel-pos
        return {"t":t, "pos":pos, "vel":vel}

env = myEnv()
env.logger = fym.Logger("data.h5")
env.reset()

while True:
    env.render()
    done=env.step()
    

    if done:
        break

env.close()
