import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from skimage.util import crop
from skimage import color
from skimage.transform import rescale
import matplotlib.pyplot as plt

def format_im(im):
    im = crop(im,((100,100),(100,100),(0,0)))
    im = -(color.rgb2gray(im)-1)
    im = rescale(im, 1.0/6.0, anti_aliasing=False)#, multichannel=False)
    return im

class ImPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=100
        self.max_torque=0.5
   
        self.viewer = None
        self.g = 10.
        self.m = 0.15
        self.l = 0.5
        self.mu = 0.05
        self.dt = 0.02
        high = np.ones(50**2+1)
        high[-1] = np.inf
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def x_next(self, x,u):
        x_theta = x[0] + x[1]*self.dt
        x_theta_dot = x[1] + (self.g/self.l*np.sin(x[0])-self.mu/(self.m*self.l**2)*x[1]+1/(self.m*self.l**2)*(u))*self.dt
        return np.array([x_theta, x_theta_dot])

    def get_velocity(self, x, u):
        u = np.clip(u,-0.5,0.5)
        x_theta = x[1]*self.dt
        x_theta_dot =  (self.g/self.l*np.sin(x[0])-self.mu/(self.m*self.l**2)*x[1]+1/(self.m*self.l**2)*(u[0]))*self.dt
        return np.array([x_theta, x_theta_dot])

    def step(self,u):
        th, thdot = self.state # th := theta  

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        #u = u[0]
        d_u = u - np.copy(self.last_u)
        self.last_u = u # for rendering
        th_normalized = angle_normalize(th)
        #print(th_normalized)
        costs = th_normalized**2 + 0.1*thdot**2 + 0.001*u**2 #+ 1*d_u**2
        
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        self.state = self.x_next(self.state, u)
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 3])
        self.state = self.np_random.uniform(low=-high, high=high)
        #self.im  = np.flatten(format_im(self.env.render(mode='rgb_array')))
        self.last_u = 0
        return self._get_obs()

    def set_state(self, x0):
        self.state = x0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        im = np.matrix.flatten(format_im(self.render(mode='rgb_array')))
        ob = np.append(im,thetadot)
        return ob

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
            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            #self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        #self.viewer.add_onetime(self.img)
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
