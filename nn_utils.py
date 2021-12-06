from numpy import genfromtxt
import numpy as np
from jax import jacobian
import jax.numpy as jnp
import torch

def get_network_func():
	W1 = jnp.asarray(genfromtxt('Wb_s32_tanh/W1.csv',delimiter=','))
	W2 = jnp.asarray(genfromtxt('Wb_s32_tanh/W2.csv',delimiter=','))
	W3 = jnp.asarray(genfromtxt('Wb_s32_tanh/W3.csv',delimiter=','))

	pi_nn_jax = lambda x : W3@jnp.tanh(W2@jnp.tanh(W1@x))
	pi_nn = lambda x : W3@np.tanh(W2@np.tanh(W1@x))
	jac_pi_nn = jacobian(pi_nn_jax, argnums=0)

	return pi_nn, pi_nn_jax, jac_pi_nn

def get_network_func_sac(env_name, policy, alpha,hidden_size):
	model = torch.load('./train_RL/sac_models/sac_actor_{}_{}_{}_{}'.format(env_name, policy, alpha, hidden_size))
	W1 = np.asarray(model['linear1.weight']).astype('float')
	W2 = np.asarray(model['linear2.weight']).astype('float')
	W3 = np.asarray(model['mean_linear.weight']).astype('float')
	x = np.array([0.1,0.1])
	y = W3@np.tanh(W2@np.tanh(W1@x))
	low = -1
	high = 1
	action_scale = (high - low) / 2.
	pi_nn = lambda x : action_scale*np.tanh(W3@np.tanh(W2@np.tanh(W1@x)))[0]

	return pi_nn