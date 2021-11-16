from numpy import genfromtxt
import numpy as np
from jax import jacobian
import jax.numpy as jnp
def get_network_func():
	W1 = jnp.asarray(genfromtxt('Wb_s32_tanh/W1.csv',delimiter=','))
	W2 = jnp.asarray(genfromtxt('Wb_s32_tanh/W2.csv',delimiter=','))
	W3 = jnp.asarray(genfromtxt('Wb_s32_tanh/W3.csv',delimiter=','))

	pi_nn = lambda x : W3@jnp.tanh(W2@jnp.tanh(W1@x))

	return pi_nn


pi_nn_fn = get_network_func()
dpi_dx = jacobian(pi_nn_fn, argnums=0)

x0 = np.array([1.0, 1.0])
y = pi_nn_fn(x0)
dy = dpi_dx(x0)
print(y)
print(dy)