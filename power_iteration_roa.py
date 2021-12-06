import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jacobian, jacfwd
#from cvxpy import Variable, Problem, Minimize, norm, quad_form
jax.config.update('jax_platform_name', 'cpu')
#integrate dynamics forward in time with input signal u with zero-order hold.

# def project_to_ellipsoid(P, r, x0):
# 	x = Variable((2,1))
# 	p = x0.reshape(2,1)
# 	constraints = []
# 	constraints.append(quad_form(x,P) <= r**2)
# 	objective = norm(p-x, p=2)
# 	prob = Problem(Minimize(objective), constraints)
# 	prob.solve('SCS')
# 	x_sol = np.asarray(x.value, dtype='float')
# 	return x_sol.reshape(2,)

def phi(f_fn, y_fn, u, T, x0):
	x_dim = x0.shape[0]
	x = np.zeros((T+1, x_dim))
	x[0,:] = x0 + u[0,:]
	x_i = x0

	for i in range(1,T+1):
		x_i = f_fn(x_i, u[i-1], i-1)
		#if i==0:
		#	print(x_i)
		x[i] = x_i
		#print(i,x[i])
	y0 = y_fn(x[0])
	
	y = np.zeros((T+1, y0.shape[0]))
	y[0] = y0
	for i in range(1,T+1):
		y[i] = y_fn(x[i])
	return x, y

#prepare inputs for adjoint
def pi(x, u, y):
	nu = y
	return x, u, y, nu

#integrate adjoint backwards in time
def psi(f_fn, y_fn, nu, 
		x, u, y, T, P):
	lam_dim = 2
	nu_dim = 2
	gamma_dim = 2
	P_inv = np.linalg.inv(P)
	Cs = np.zeros((T, gamma_dim, lam_dim))
	Ds = np.zeros((T, gamma_dim, nu_dim))

	lam = np.zeros((T+1, lam_dim))

	#dfdx = jacobian(f_fn,argnums=0)
	#dfdu = jacobian(f_fn,argnums=1)

	# dydx = jacobian(y_fn,argnums=0)
	# dydu = jacobian(y_fn,argnums=1)
	# dfdx = jacfwd(f_fn, argnums=0)
	# dfdu = jacfwd(f_fn, argnums=1)
	
	lam0 = y[T]
	lam[T] = lam0
	#print(lam[T])
	for i in range(0,T):
		t = (T-1) - i
		#print(t)
		# dfdx_ = dfdx(x[t], u[t], t).T
		# dfdu_ = dfdu(x[t], u[t], t).T
		dfdx_ = jacfwd(lambda x : f_fn(x, u[t], t))(x[t]).T
		dfdu_ = jacfwd(lambda u : f_fn(x[t], u, t))(u[t]).T
		# dfdx_ = dfdx(x[t], u[t], t).T
		# dfdu_ = dfdu(x[t], u[t], t).T
		#dydx_ = 2*dydx(x_rev[i], u_rev[i]).T
		#dydu_ = 2*dydu(x_rev[i], u_rev[i]).Tja

		A = dfdx_#(x_rev[i], u_rev[i], w_rev[i], delta_u).T
		B = np.zeros((2,2))#2*dydx_
		C = dfdu_
		D = np.zeros((2,2))#2*dydu_
		Cs[t] = C
		Ds[t] = D

		#if i < T-1:
		lam[t] = f_x_linear(A, B, lam[t+1], nu[t])
		#print(t,lam[t])
	#print('end')
	#gamma0 = y_x_linear(Cs[0], Ds[0], lam[0], nu[0])
	gamma = np.zeros((T,gamma_dim))
	#gamma[0] = gamma0
	#print('gammas')
	for i in range(0,T):
		gamma[i] = -P_inv@y_x_linear(Cs[i], Ds[i], lam[i+1], nu[i])
		#print(i, gamma[i])
	#plt.plot([0,gamma[0,0]],[0,gamma[0,1]])
	#plt.axis('equal')
	#plt.show()
	return np.asarray(gamma)

# renormalize next inputs
def theta(gamma, R, P):
	gamma_norm = L2_norm(gamma, P)

	#if gamma_norm < 1e-11:
	#	u_next = gamma*0
	#else:
	#print('init norm',gamma_norm*np.sqrt(500))
	#if gamma_norm > R:
	#	u_0 = project_to_ellipsoid(P,R,gamma[0])
	#	gamma[0] = u_0
	#	u_ = gamma
	#	#print('here')
	#else:
	u_ = gamma * R / gamma_norm
	#print(L2_norm(u_,P)*np.sqrt(500))
	u_next = u_
	return u_next

#numerically integrate L2 norm of discrete signal function. sqrt(int^T_0 y^T y dt) 
def L2_norm(y, P):
	#P = np.asarray([[8,0],[0,1]])
	if len(y.shape) == 1:
		y = np.expand_dims(y,axis=-1)
	#y_square = np.linalg.norm(y,2,axis=1)**2
	y_square_sum = 0

	for i in range(y.shape[0]):
		y_i = y[i].reshape(2,1)
		y_square_sum += y_i.T@P@y_i
		#print('y shape',y.shape[0])
	return np.sqrt(y_square_sum)
	#print(y_square)
	#return np.sqrt(1/y.shape[0]*np.sum(y_square))

def f_x_linear(A,B,x,u):
	return A@x + B@u

def y_x_linear(C,D,x,u):
	return C@x + D@u

def weighting_P(theta, major_ax, minor_ax):
	R = np.array([[np.cos(theta), -np.sin(theta)],
				[np.sin(theta), np.cos(theta)]])
	P_ = np.array([[major_ax, 0],[0, minor_ax]])
	P = R.T@P_@R
	return P
