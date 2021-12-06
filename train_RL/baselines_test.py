import gym
import gym_custom
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

env = gym.make("ImPendulum-v0")

model = SAC.load("im_pen_model_10000_steps")
#print(model.get_parameters()['policy'])
Z = []
obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    Z.append(env.env.state)

Z = np.asarray(Z)
plt.plot(Z[:,0],label='theta')
plt.plot(Z[:,1],label='theta dot')
plt.show()