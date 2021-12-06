import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from power_iteration_discrete import *
from tqdm import tqdm
from utils import get_network_func


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
# def main():
# 	N = 30
# 	R = 1.0
# 	x0 = np.array([.0, .0])
# 	lam0 = np.zeros(x0.shape[0])
# 	u = np.random.uniform(-1,1,size=N)
# 	t = np.linspace(0,10,N)
# 	u = u*R/L2_norm(u.reshape(N,1),t)
# 	plt.plot(t,u)
# 	for i in tqdm(range(30)):
# 		x, y = phi(f_x, y_x, u, t, x0)
# 		x_rev, u_rev, nu, t_nu = pi(x,u,y,t)
# 		lam, gamma = psi(f_x_jax, y_x, nu, x_rev, u_rev, t_nu, lam0)
# 		u = theta(gamma, t_nu, R).reshape(N,)
# 		#print(i)
# 		plt.plot(t,u)
# 	print('Computed Worst-cas Gain: {}'.format(L2_norm(y,t)/L2_norm(u.reshape(N,1),t)))
# 	plt.ylabel('u(t)')
# 	plt.xlabel('t')
# 	plt.grid()
# 	plt.show()
def main():
	c1='y' #blue
	c2='red' #green

	T = 1000
	R = 1.0
	x0 = np.array([0.0, 0.0])
	lam0 = np.zeros(x0.shape[0])
	lam_w = 1.0
	u = np.random.uniform(-1,1,size=T)
	w = np.random.uniform(-1,1,size=T)
	#t = np.linspace(0,10,N)
	u = u*R/L2_norm(u.reshape(T,1))
	plt.plot(u, c=colorFader(c1,c2,0))
	iterations = 20
	pi_nn, pi_nn_jax, jac_pi_nn = get_network_func()
	delta_u = 0.1
	delta_z = 0.1

	g = 10.
	m = 0.15
	l = 0.5
	mu = 0.05
	dt = 0.02


	def f_pen(x, u):
		return np.array([x[0] + x[1]*dt, x[1] + (g/l*np.sin(x[0])-mu/(m*l**2)*x[1]+1/(m*l**2)*(pi_nn(x) + u))*dt])

	def f_pen_jax(x, u):
		return jnp.array([x[0] + x[1]*dt, x[1] + (g/l*jnp.sin(x[0])-mu/(m*l**2)*x[1]+1/(m*l**2)*(pi_nn_jax(x) + u))*dt])

	def y_pen(x, u):
		return np.array([x[0],x[1],pi_nn(x)*0.1])

	def y_pen_jax(x, u):
		return jnp.array([x[0],x[1],pi_nn_jax(x)*0.1])

	for i in tqdm(range(iterations)):
		x, y, z = phi(f_x_1, y_x_1, z_x_1, u, w, delta_u, delta_z, T, x0)
		x_rev, u_rev, w_rev, nu, eta = pi(x,u,w,y,z,lam_w)
		gamma, kappa, lam_i = psi(f_x_1_jax, y_x_1_jax, z_x_1_jax, 
							nu, eta, 
							x_rev, u_rev, w_rev, 
							delta_u, delta_z, T, lam0)
		u, w, lam_w = theta(gamma, kappa, w, z, lam_w, R)
		delta_u, delta_z = update_params(lam_i, delta_u, delta_z)
		# delta_u = 1.0
		# delta_z = 1.0
		print(L2_norm(z)/L2_norm(w))
		plt.plot(u,c=colorFader(c1,c2,(i+1)/iterations))
		x,y,z=phi(f_x_1, y_x_1, z_x_1, u, w, delta_u, delta_z, T, x0)
		z_y = np.concatenate((z,y),axis=1)
		u_w = np.concatenate((u,w),axis=1)
		print('lam_i', lam_i)
		print('deltas: ',delta_u, delta_z)
		print('Computed Worst-cas Gain: {}'.format(L2_norm(z_y)/L2_norm(u_w.reshape(T,2))))
	plt.ylabel('u(t)')
	plt.xlabel('t')
	plt.grid()
	plt.show()

	plt.plot(u)
	plt.show()


if __name__ == "__main__":
	main()