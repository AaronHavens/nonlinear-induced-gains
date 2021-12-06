import gym
import gym_custom
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make("ImPendulum-v0")

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./',
                                         name_prefix='im_pen_model')
model = SAC("MlpPolicy", env, buffer_size=10000, verbose=1)
model.learn(total_timesteps=20000, log_interval=4, callback=checkpoint_callback)
model.save("sac_pendulum")