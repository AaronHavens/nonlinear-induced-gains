import gym
from skimage import color
from skimage.transform import rescale
import numpy as np

def format_im(im):
    im = -(color.rgb2gray(im)-1)
    im = rescale(im, 1.0/5.0, anti_aliasing=False, multichannel=False)
    return im


class ImageEnv():
    def __init__(self, env):

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, a):
        ob,r,done,info = self.env.step(a)
        new_im_ob = format_im(self.env.render(mode='rgb_array'))
        im_ob = np.stack((self.last_im, new_im_ob),axis=-1)# 2 channel greyscale image
        self.last_im = new_im_ob

        return im_ob, r, done, info

    def reset(self):
        self.env.reset()
        self.last_im  = format_im(self.env.render(mode='rgb_array'))
        im_ob,_,_,_ = self.step(np.zeros(self.action_space.sample().shape))
        return im_ob

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

