import gym
import gym_custom
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from sac import SAC
import argparse
from collections import deque
import copy
import sys
sys.path.append('../')
from nn_utils import get_network_func
def dare(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--flex', action="store_true")
parser.add_argument('--filter', action="store_true")
args = parser.parse_args()


#K = dare(Af, Bf, Q, R)

#xhat = np.array([x[0],0,x[1],0])
#env.state = np.array([1,0,-1,0])
#env.state = np.array([1,3])
#x = np.array([1,-1])
N = 500
X = []
Y = []
V = []
R = 0

env = gym.make(args.env_name)
#env.seed(args.seed)

x = env.reset()
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model('./sac_models/sac_actor_{}_{}_{}_{}'.format(args.env_name, args.policy, args.alpha, args.hidden_size), './sac_models/sac_critic_{}_{}_{}_{}'.format(args.env_name, args.policy, args.alpha, args.hidden_size))
pi_nn, pi_nn_jax, jac_pi_nn = get_network_func()
Z = []

for i in range(N):
    X.append(x[0])
    Y.append(x[1])
    V.append(x[2])
    u = agent.select_action_mean(x)
    #u = np.array([np.clip(pi_nn(x),-.7, .7)])
    #print(u)
    x,r,done,_ = env.step(u)
    state = env.env.state
    print(np.linalg.norm(state,2))
    env.env.render()
    Z.append(x)
    R += r

Z = np.array(Z)
#print(Z.shape)
env.close()

t = 1e-1*np.arange(0,N)
t1 = 1e-1*np.arange(0,N)
#print(agent.policy_old.mean.weight, K)
plt.xlabel('time [s]')
plt.title('Test')
#plt.plot(t,X2, label='position weighted LQR',c='black',linestyle='--')
#plt.plot(t,Y2, label='velocity weighted LQR',c='gray', linestyle='--')
#plt.plot(t1,Z2[:,0], label='TD3 delay=0 [s] position', c='g')
#plt.plot(t1,Z2[:,1], label='TD3 delay=0 [s] velocity', c='m')
plt.plot(t,Z[:,0], label='SAC x',c='orange')
plt.plot(t,Z[:,1], label='SAC y',c='c')
plt.plot(t,Z[:,2], label='SAC theta dot', c='r')
#plt.plot(t1,Z1[:,0], label='TD3 delay=1e-3 [s] position', c='r', linestyle='-')
#plt.plot(t1,Z1[:,1], label='TD3 delay=1e-3 [s] velocity', c='b', linestyle='-')
#plt.ylim(-3,3)
plt.legend()
plt.show()
#plt.plot(V, label='velocity v')
