import numpy as np
import matplotlib.pyplot as plt
from power_iteration import *
from tqdm import tqdm

def main():
	N = 30
	R = 1.0
	x0 = np.array([1.0, -1.0])
	lam0 = np.zeros(x0.shape[0])
	u = np.random.uniform(-1,1,size=N)
	t = np.linspace(0,10,N)
	u = u*R/L2_norm(u.reshape(N,1),t)
	plt.plot(t,u)
	for i in tqdm(range(30)):
		x, y = phi(f_x, y_x, u, t, x0)
		x_rev, u_rev, nu, t_nu = pi(x,u,y,t)
		lam, gamma = psi(f_x_jax, y_x, nu, x_rev, u_rev, t_nu, lam0)
		u = theta(gamma, t_nu, R).reshape(N,)
		#print(i)
		plt.plot(t,u)
	print('Computed Worst-cas Gain: {}'.format(L2_norm(y,t)/L2_norm(u.reshape(N,1),t)))
	plt.ylabel('u(t)')
	plt.xlabel('t')
	plt.grid()
	plt.show()


if __name__ == "__main__":
	main()