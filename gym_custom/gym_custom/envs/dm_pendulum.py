import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from dm_control import suite

def rgb2gray(rgb):
    print(rgb.shape)
    rgb_crop = crop(rgb, ((70,70),(50,0), (0,0)), copy=False)
    rgb_resize = resize(rgb, (64,64,3))
    return np.dot(rgb_resize[...,:3], [0.2989, 0.5870, 0.1140])

class PendulumDmEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.viewer = None
        self.dm_env = suite.load(domain_name="pendulum", task_name="swingup")
        action_spec = self.dm_env.action_spec()
        obs_spec = self.dm_env.observation_spec()
        high = np.array([1,1,np.inf])
        self.max_torque=action_spec.maximum
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta  
        #u = np.clip(u, -0.7, 0.7)
        time_step = self.dm_env.step(u)
        reward = -(th**2 + 0.1*thdot**2 + 0.001*u**2)[0]
        orientation = time_step.observation['orientation']
        #print(orientation)
        velocity = time_step.observation['velocity']
        theta = np.arctan2(orientation[1],orientation[0])
        #(theta)
        self.last_u = u
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        self.state = np.array([theta, velocity[0]])
        return self.get_obs(orientation, velocity), reward, False, {}

    def reset(self):
        time_step = self.dm_env.reset()
        orientation = time_step.observation['orientation']
        velocity = time_step.observation['velocity']
        theta = np.arctan2(orientation[1],orientation[0])
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        #print('vel',velocity)
        self.state = np.array([theta, velocity[0]])
        self.last_u = None
        return self.get_obs(orientation, velocity)
    
    def get_obs(self, orientation, velocity):
        return np.array([orientation[0], orientation[1], velocity[0]])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
