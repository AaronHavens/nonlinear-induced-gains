import gym
import gym_custom
import numpy as np
from gym_image_wrap import ImageEnv
import matplotlib.pyplot as plt

env = gym.make('ImPendulum-v0')
#im_env = ImageEnv(env)
x = env.reset()

for i in range(100):
	a = env.action_space.sample()
	x,r,done,_ = env.step(a)
	print(x.shape)
	plt.imshow(x)
	plt.show()
	#env.render()
