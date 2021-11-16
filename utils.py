from numpy import genfromtxt
import numpy as np
from jax import jacobian
import jax.numpy as jnp

def get_network_func():
	W1 = jnp.asarray(genfromtxt('Wb_s32_tanh/W1.csv',delimiter=','))
	W2 = jnp.asarray(genfromtxt('Wb_s32_tanh/W2.csv',delimiter=','))
	W3 = jnp.asarray(genfromtxt('Wb_s32_tanh/W3.csv',delimiter=','))

	pi_nn_jax = lambda x : W3@jnp.tanh(W2@jnp.tanh(W1@x))
	pi_nn = lambda x : W3@np.tanh(W2@np.tanh(W1@x))
	jac_pi_nn = jacobian(pi_nn_jax, argnums=0)

	return pi_nn, pi_nn_jax, jac_pi_nn
