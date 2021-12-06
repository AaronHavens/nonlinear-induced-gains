import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jacobian
jax.config.update('jax_platform_name', 'cpu')
#integrate dynamics forward in time with input signal u with zero-order hold.
def phi(f_fn, y_fn, z_fn, u, v, delta_u, delta_z, T, x0):
	x_dim = x0.shape[0]
	x = np.zeros((T, x_dim))
	x[0,:] = x0
	for i in range(1,T):
		x_i = f_fn(x[i-1,:], u[i-1], v[i-1], delta_u)
		x[i] = x_i

	y0 = y_fn(x[0],u[0], v[0])
	z0 = z_fn(x[0],u[0], v[0], delta_z)
	y = np.zeros((T, y0.shape[0]))
	z = np.zeros((T, z0.shape[0]))
	y[0] = y0
	for i in range(1,T):
		y[i] = y_fn(x[i], u[i], v[i])
		z[i] = z_fn(x[i], u[i], v[i], delta_z)
	
	return x, y, z

#prepare inputs for adjoint
def pi(x, u, w, y, z, lambda_w):
	return x[::-1], u[::-1], w[::-1], y[::-1], lambda_w*z[::-1]

#integrate adjoint backwards in time
def psi(f_fn, y_fn, z_fn, nu, eta, 
		x_rev, u_rev, w_rev, 
		delta_u, delta_z, T, lam0):
	Cs = []
	Ds = []
	lam_dim = lam0.shape[0]
	nu_dim = nu.shape[1]
	kappa_dim = 1
	eta_dim = 1
	gamma_dim = 1
	#print(lam_dim, nu_dim, eta_dim, gamma_dim)
	lam = np.zeros((T, lam_dim+2))
	#lam[0] = np.concatenate((lam0, np.array([delta_u,delta_z])),axis=0)
	#lam[0] = lam0
	#print(lam[0])
	dfdx = jacobian(f_fn,argnums=0)
	dfdu = jacobian(f_fn,argnums=1)
	dfdw = jacobian(f_fn,argnums=2)
	dfddu = jacobian(f_fn,argnums=3)
	dfddz = 0

	dydx = jacobian(y_fn,argnums=0)
	dydu = jacobian(y_fn,argnums=1)
	dydw = jacobian(y_fn,argnums=2)
	dyddu = 0
	dyddz = 0

	dzdx = jacobian(z_fn,argnums=0)
	dzdu = jacobian(z_fn,argnums=1)
	dzdw = jacobian(z_fn,argnums=2)
	dzddu = 0
	dzddz = jacobian(z_fn,argnums=3)

	# should check shape for null dim 1.
	#print(nu.shape,eta.shape)
	nu_eta = np.concatenate((nu,eta),axis=1)
	
	for i in range(T):
		dfdx_ = dfdx(x_rev[i], u_rev[i], w_rev[i], delta_u).T
		dfdu_ = dfdu(x_rev[i], u_rev[i], w_rev[i], delta_u).T
		dfdw_ = dfdw(x_rev[i], u_rev[i], w_rev[i], delta_u).T
		dfddu_ = dfddu(x_rev[i], u_rev[i], w_rev[i], delta_u).T
		dfddz_ = np.array([[0,0]])

		dydx_ = 2*dydx(x_rev[i], u_rev[i], w_rev[i]).T
		dydu_ = 2*dydu(x_rev[i], u_rev[i], w_rev[i]).T
		dydw_ = 2*dydw(x_rev[i], u_rev[i], w_rev[i]).T
		dyddu_ = np.array([[0,0,0]])
		dyddz_ = np.array([[0,0,0]])

		dzdx_ = dzdx(x_rev[i], u_rev[i], w_rev[i], delta_z).T
		dzdu_ = dzdu(x_rev[i], u_rev[i], w_rev[i], delta_z).T
		dzdw_ = dzdw(x_rev[i], u_rev[i], w_rev[i], delta_z).T
		dzddu_ = 0
		dzddz_ = dzddz(x_rev[i], u_rev[i], w_rev[i], delta_z).T

		# print('A dims')
		# print(dfdx_.shape)
		#print(dfddu_, dfddz_)
		A = np.block([[dfdx_, np.zeros((2,2))],
					[dfddu_, 1,0],[dfddz_,0,1]])
		# print(A.shape)
		#print('B dims')
		# print(dydx_.shape, dzdx_.shape)
		# print(dyddu_, dzddu_)
		# print(dyddz_, dzddz_)
		B = np.block([[dydx_, dzdx_],
					[dyddu_,dzddu_],
					[dyddz_,dzddz_]])
		# print(B.shape)
		# print('C dims')
		# print(dfddu_.shape, dfdw_.shape)
		C = np.block([[dfdu_, 0,0],
						[dfdw_,0,0]])
		# print(C.shape)
		# print('D dims')
		# print(dydu_.shape, dzdu_.shape)
		# print(dydw_.shape, dzdw_.shape)
		D = np.block([[dydu_,dzdu_],
						[dydw_,dzdw_]])

		# A = dfdx_#(x_rev[i], u_rev[i], w_rev[i], delta_u).T
		# B = np.block([[2*dydx_, dzdx_]])
		
		# C = np.block([[dfdu_],
		# 				[dfdw_]])
		# D = np.block([[2*dydu_, dzdu_],
		# 				[2*dydw_,dzdw_]])
		# #print(C, D)
		Cs.append(C)
		Ds.append(D)
		#print(A.shape, B.shape, C.shape, D.shape)
		#lam_ext = np.concatenate((lam[i], np.array([delta_u,delta_z])))
		if i < T-1:
			lam[i+1] = f_x_linear(A, B, lam[i], nu_eta[i])
			#print(lam[i+1,-2:])
	#print('end')
	adjoint_out0 = y_x_linear(Cs[0], Ds[0], lam[0], nu_eta[0])
	gamma0 = adjoint_out0[:gamma_dim]
	kappa0 = adjoint_out0[gamma_dim:]
	gamma = np.zeros((T,gamma_dim))
	kappa = np.zeros((T,1))
	gamma[0] = gamma0
	kappa[0] = kappa0
	for i in range(1,T):
		adjoint_out = y_x_linear(Cs[i], Ds[i], lam[i], nu_eta[i])
		gamma[i] = adjoint_out[:gamma_dim]
		kappa[i] = adjoint_out[gamma_dim:]
	#print(lam[-1,-2:])
	return np.asarray(gamma[::-1]), np.asarray(kappa[::-1]), np.asarray(lam[-1,-2:])

# renormalize next inputs
def theta(gamma, kappa, w, z, lambda_w, R):
	gamma_norm = L2_norm(gamma)
	kappa_norm = L2_norm(kappa)
	lambda_w = 3*L2_inner(kappa, w)/(L2_norm(z)*L2_norm(w))
	if gamma_norm < 1e-4:
		u_next = gamma*0
	else:
		u_ = gamma * R / gamma_norm
		u_next = u_[::-1]
	w_next = 1/lambda_w*kappa[::-1]
	return u_next, w_next, lambda_w

def update_params(lam_i, delta_u, delta_z):
	delta_u = np.clip(delta_u+lam_i[0], 0.01, 1)
	delta_z = np.clip(delta_z+lam_i[1], 0.01, 1)
	return delta_u, delta_z

#numerically integrate L2 norm of discrete signal function. sqrt(int^T_0 y^T y dt) 
def L2_norm(y):
	if len(y.shape) == 1:
		y = np.expand_dims(y,axis=-1)
	y_square = np.linalg.norm(y,2,axis=1)**2
	#print(y_square)
	return np.sqrt(1/y.shape[0]*np.sum(y_square))

def L2_inner(x1,x2):
	Sn = 0
	T = x1.shape[0]
	for i in range(T):
		Sn += np.inner(x1[i],x2[i])
	return Sn/T

def f_x_linear(A,B,x,u):
	return A@x + B@u

def y_x_linear(C,D,x,u):
	return C@x + D@u

def f_x(x_k,u_k):

	return np.ones(2) + 0.1*np.array([-x_k[0] + x_k[1] - x_k[0]*x_k[1]**2,-x_k[1]*x_k[0]**2 - x_k[1] + u_k])

def f_x_1_jax(x_k, u_k, v_k, delta_u):
	A = jnp.array([[1.0, 0.1],[0.0, 1.0]])
	B = jnp.array([[0.0],[0.1]])
	K = jnp.array([[-2.0,-2.0]])
	return (A+B@K)@x_k + B@(delta_u*u_k.reshape(1,)+0.25*v_k.reshape(1,))

def f_x_1(x_k, u_k, v_k, delta_u):
	A = np.array([[1.0, 0.1],[0.0, 1.0]])
	B = np.array([[0.0],[0.1]])
	K = np.array([[-2.0,-2.0]])
	return (A+B@K)@x_k + B@(delta_u*u_k.reshape(1,)+0.25*v_k.reshape(1,))

def y_x_1(x_k, u_k, v_k):
	K = np.array([[-2.0,-2.0]])
	pi_x = K@x_k
	return np.array([x_k[0],x_k[1], pi_x[0]*0.1])

def y_x_1_jax(x_k, u_k, v_k):
	K = jnp.array([[-2.0,-2.0]])
	pi_x = K@x_k
	return jnp.array([x_k[0],x_k[1], pi_x[0]*0.1])

def z_x_1(x_k, u_k, v_k, delta_z):
	return (delta_z*np.array([-2.0, -2.0])@x_k).reshape(1,)

def z_x_1_jax(x_k, u_k, v_k, delta_z):
	return (delta_z*jnp.array([-2.0, -2.0])@x_k).reshape(1,)


def f_x_jax(x_k,u_k):

	return jnp.ones(2) + 0.1*jnp.array([-x_k[0] + x_k[1] - x_k[0]*x_k[1]**2,-x_k[1]*x_k[0]**2 - x_k[1] + u_k])

def y_x(x_k, u_k):
	return x_k


