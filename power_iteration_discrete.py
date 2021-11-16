import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jacobian
jax.config.update('jax_platform_name', 'cpu')
#integrate dynamics forward in time with input signal u with zero-order hold.
def phi(f_fn, y_fn, u, T, x0):
	x_dim = x0.shape[0]
	x = np.zeros((T, x_dim))
	x[0,:] = x0
	for i in range(1,T):
		x_i = f_fn(x[i,:], u[i])
		x[i] = x_i

	y0 = y_fn(x[0],u[0])
	y = np.zeros((T, y0.shape[0]))
	y[0] = y0
	for i in range(1,T):
		y[i] = y_fn(x[i], u[i])
	
	return x, y 

#prepare inputs for adjoint
def pi(x, u, y):
	return x[::-1], u[::-1], y[::-1]

#integrate adjoint backwards in time
def psi(f_fn, y_fn, nu, x_rev, u_rev, T, lam0):
	Cs = []
	Ds = []
	lam_dim = lam0.shape[0]
	nu_dim = nu.shape[1]
	lam = np.zeros((T, lam_dim))
	lam[0] = lam0

	dfdx = jacobian(f_fn,argnums=0)
	dfdu = jacobian(f_fn,argnums=1)
	dydx = jacobian(y_fn,argnums=0)
	dydu = jacobian(y_fn,argnums=1)

	for i in range(T):

		A = dfdx(x_rev[i], u_rev[i]).T
		B = 2*dydx(x_rev[i],u_rev[i]).T
		
		C = (dfdu(x_rev[i], u_rev[i]).reshape(lam_dim,1)).T
		D = (2*dydu(x_rev[i], u_rev[i]).reshape(nu_dim,1)).T
		#print(C, D)
		Cs.append(C)
		Ds.append(D)
		if i < T-1:
			lam_i = f_x_linear(A, B, lam[i,:], nu[i])
			lam[i+1] = lam_i
	gamma0 = y_x_linear(Cs[0], Ds[0], lam[0], nu[0])
	gamma = np.zeros((T,gamma0.shape[0]))
	gamma[0] = gamma0
	for i in range(1,T):
		gamma[i] = y_x_linear(Cs[i], Ds[i], lam[i], nu[i])
	return np.asarray(lam), np.asarray(gamma)

# renormalize next inputs
def theta(gamma, R):
	gamma_norm = L2_norm(gamma)
	if gamma_norm < 1e-4:
		u_next = gamma*0
	else:
		u_ = gamma * R / gamma_norm
		u_next = u_[::-1]
	return u_next


#numerically integrate L2 norm of discrete signal function. sqrt(int^T_0 y^T y dt) 
def L2_norm(y):
	y_square = np.linalg.norm(y,2,axis=1)

	return np.sqrt(1/y.shape[0]*np.sum(y_square))

def f_x_linear(A,B,x,u):
	return A@x + B@u

def y_x_linear(C,D,x,u):
	return C@x + D@u


def f_x(x_k,u_k):

	return np.ones(2) + 0.1*np.array([-x_k[0] + x_k[1] - x_k[0]*x_k[1]**2,-x_k[1]*x_k[0]**2 - x_k[1] + u_k])

def f_x_1_jax(x_k, u_k):
	A = jnp.array([[1.0, 0.1],[0.0, 1.0]])
	B = jnp.array([[0.0],[0.1]])
	K = jnp.array([[-2.0,-2.0]])
	return (A+B@K)@x_k + B@(u_k.reshape(1,))

def f_x_1(x_k, u_k):
	A = np.array([[1.0, 0.1],[0.0, 1.0]])
	B = np.array([[0.0],[0.1]])
	K = np.array([[-2.0,-2.0]])
	return (A+B@K)@x_k + B@(u_k.reshape(1,))

def f_x_jax(x_k,u_k):

	return jnp.ones(2) + 0.1*jnp.array([-x_k[0] + x_k[1] - x_k[0]*x_k[1]**2,-x_k[1]*x_k[0]**2 - x_k[1] + u_k])

def y_x(x_k, u_k):
	return x_k


