import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from power_iteration_discrete import *
from tqdm import tqdm

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
	R = 10.0
	x0 = np.array([.0, .0])
	lam0 = np.zeros(x0.shape[0])
	u = np.random.uniform(-1,1,size=T)
	#t = np.linspace(0,10,N)
	u = u*R/L2_norm(u.reshape(T,1))
	plt.plot(u, c=colorFader(c1,c2,0))
	for i in tqdm(range(10)):
		x, y = phi(f_x_1, y_x, u, T, x0)
		x_rev, u_rev, nu = pi(x,u,y)
		lam, gamma = psi(f_x_1_jax, y_x, nu, x_rev, u_rev, T, lam0)
		u = theta(gamma, R).reshape(T,)
		#print(i)
		plt.plot(u,c=colorFader(c1,c2,(i+1)/10))
		print('Computed Worst-cas Gain: {}'.format(L2_norm(y)/L2_norm(u.reshape(T,1))))
	plt.ylabel('u(t)')
	plt.xlabel('t')
	plt.grid()
	plt.show()


if __name__ == "__main__":
	main()