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

	T = 200
	R = 1.0
	x0 = np.array([0.0, 0.0])
	lam0 = np.zeros(x0.shape[0])
	u = np.random.uniform(-1,1,size=T)
	#t = np.linspace(0,10,N)
	u = u*R/L2_norm(u.reshape(T,1))
	plt.plot(u, c=colorFader(c1,c2,0))
	iterations = 50
	pi_nn, pi_nn_jax, jac_pi_nn = get_network_func()

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
		x, y = phi(f_x_1, y_x_1, u, T, x0)
		x_rev, u_rev, nu = pi(x,u,y)
		lam, gamma = psi(f_x_1_jax, y_x_1_jax, nu, x_rev, u_rev, T, lam0)
		u = theta(gamma, R).reshape(T,)
		#print(i)
		plt.plot(u,c=colorFader(c1,c2,(i+1)/iterations))
		print('Computed Worst-cas Gain: {}'.format(L2_norm(y)/L2_norm(u.reshape(T,1))))
	plt.ylabel('u(t)')
	plt.xlabel('t')
	plt.grid()
	plt.show()

	plt.plot(u)
	plt.show()


if __name__ == "__main__":
	main()