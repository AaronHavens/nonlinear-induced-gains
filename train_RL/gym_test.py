from dm_control import suite
from dm_control.suite.wrappers import pixels
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import crop
# Load one task:

def rgb2gray(rgb):
	print(rgb.shape)
	rgb_crop = crop(rgb, ((70,70),(50,0), (0,0)), copy=False)
	rgb_resize = resize(rgb, (64,64,3))
	return np.dot(rgb_resize[...,:3], [0.2989, 0.5870, 0.1140])

env = suite.load(domain_name="pendulum", task_name="swingup")
env_only_pixels = pixels.Wrapper(env, pixels_only=False,  render_kwargs={'camera_id': 0})
# Step through an episode and print out reward, discount and observation.
action_spec = env_only_pixels.action_spec()
obs_spec = env_only_pixels.observation_spec()
time_step = env.reset()
print(obs_spec)
print(action_spec)
while not time_step.last():
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)
  print(time_step)
  print(env._task.get_reward(env._physics))
  #gray = rgb2gray(time_step.observation['pixels'])
  #print(time_step.observation['pixels'])
  #plt.imshow(gray,cmap=plt.get_cmap('gray'))
  #plt.show()
  #print(time_step.observation['orientation'], time_step.observation['velocity'])