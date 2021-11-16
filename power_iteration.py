import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jacobian
jax.config.update('jax_platform_name', 'cpu')
#integrate dynamics forward in time with input signal u with zero-order hold.
def phi(f_fn, y_fn, u, t, x0):
	N = len(t)
	x_dim = x0.shape[0]
	x = np.zeros((N, x_dim))
	x[0,:] = x0
	for i in range(len(t)-1):
		f_fn_u = lambda t_k, x_k : f_fn(x_k, t_k, u[i])
		t_span = [t[i], t[i+1]]
		x_i = integrate.solve_ivp(f_fn_u, t_span, x[i,:])
		x_i = np.array(x_i.y)[:,-1]
		x[i+1] = x_i

	y0 = y_fn(x[0],u[0],t[0])
	y = np.zeros((N, y0.shape[0]))
	y[0] = y0
	for i in range(1,len(t)):
		y[i] = y_fn(x[i], u[i], t[i])
	
	return x, y 

#prepare inputs for adjoint
def pi(x, u, y, t):
	t_flip = np.max(t) - t[::-1]
	return x[::-1], u[::-1], y[::-1], t_flip

#integrate adjoint backwards in time
def psi(f_fn, y_fn, nu, x_rev, u_rev, t, lam0):
	N = len(t)
	Cs = []
	Ds = []
	lam_dim = lam0.shape[0]
	nu_dim = nu.shape[1]
	lam = np.zeros((N, lam_dim))
	lam[0] = lam0

	dfdx = jacobian(f_fn,argnums=0)
	dfdu = jacobian(f_fn,argnums=2)
	dydx = jacobian(y_fn,argnums=0)
	dydu = jacobian(y_fn,argnums=1)

	for i in range(N):

		A = dfdx(x_rev[i],t[i], u_rev[i]).T
		B = 2*dydx(x_rev[i],u_rev[i],t[i]).T
		
		C = (dfdu(x_rev[i], u_rev[i],t[i]).reshape(lam_dim,1)).T
		D = (2*dydu(x_rev[i], u_rev[i],t[i]).reshape(nu_dim,1)).T
		#print(C, D)
		Cs.append(C)
		Ds.append(D)
		if i < N-1:
			t_span = [t[i], t[i+1]]
			f_fn_u = lambda t_k, x_k : f_x_linear(A, B, x_k, nu[i])
			lam_i = integrate.solve_ivp(f_fn_u, t_span, lam[i,:])
			lam_i = np.array(lam_i.y)[:,-1]
			lam[i+1] = lam_i
	gamma0 = y_x_linear(Cs[0], Ds[0], lam[0], nu[0])
	gamma = np.zeros((N,gamma0.shape[0]))
	gamma[0] = gamma0
	for i in range(1,N):
		gamma[i] = y_x_linear(Cs[i], Ds[i], lam[i], nu[i])
	return np.asarray(lam), np.asarray(gamma)

# renormalize next inputs
def theta(gamma, t_rev, R):
	gamma_norm = L2_norm(gamma, t_rev)
	if gamma_norm < 1e-4:
		u_next = gamma*0
	else:
		u_ = gamma * R / gamma_norm
		u_next = u_[::-1]
	return u_next


#numerically integrate L2 norm of discrete signal function. sqrt(int^T_0 y^T y dt) 
def L2_norm(y,t):
	y_square = np.linalg.norm(y,2,axis=1)
	#print(y_square)
	y_diff = np.diff(y_square,n=1)
	#print(y_diff)
	t_diff = np.diff(t, n=1)
	S = 0
	for i in range(len(t)-1):
		S += (y_square[i]**2 + y_square[i]*y_diff[i] + y_diff[i]**2/3)*t_diff[i]

	return np.sqrt(S)

def f_x_linear(A,B,x,u):
	return np.asarray(A@x + B@u)

def y_x_linear(C,D,x,u):
	return C@x + D@u


def f_x(x_k,t_k,u_k):

	return np.array([-x_k[0] + x_k[1] - x_k[0]*x_k[1]**2,-x_k[1]*x_k[0]**2 - x_k[1] + u_k])

def f_x_jax(x_k,t_k,u_k):

	return jnp.array([-x_k[0] + x_k[1] - x_k[0]*x_k[1]**2,-x_k[1]*x_k[0]**2 - x_k[1] + u_k])

def y_x(x_k, u_k, t_k):
	return x_k + u_k*0.1


