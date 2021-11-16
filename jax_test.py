import jax.numpy as jnp
from jax import jacobian
import numpy as np



def Ax(A, x):
	return A@x


A = np.array([[2,3],[4,1]])

dAdx = jacobian(Ax,argnums=1)
x0 = np.array([1.0,1.0])
print(np.asarray(dAdx(A, x0)))
