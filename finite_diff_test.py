from power_iteration_roa import *
import numpy as np
from nn_utils import get_network_func, get_network_func_sac
from tqdm import tqdm
from ellipsoid_utils import *

def monte_carlo_ROA(f, T, N, eps=1e-3):
	x = np.linspace(-3.0, 3.0, N)
	y = np.linspace(-3.0, 3.0, N)
	xv, yv = np.meshgrid(x, y)
	ROA = np.zeros((N,N))

	for i in tqdm(range(N)):
		for j in range(N):
			x_t = np.array([xv[i,j], yv[i,j]])
			for t in range(T):
				x_t = f(x_t,np.zeros(2),1)
			x_t_norm = np.linalg.norm(x_t,2)
			#print(x_t_norm)
			is_stable = x_t_norm <= eps
			ROA[i,j] = is_stable
		
	#out = np.column_stack((xv.ravel(), yv.ravel(), ROA.ravel()))
	#return out

	return xv, yv, ROA

def P_norm(x_, P):
	x = x_.reshape(x_.shape[0],1)
	Pnorm = np.sqrt(x.T@P@x)
	return Pnorm[0,0]

def single_run(f, T, x0):
	x_t = np.copy(x0)
	for i in range(1,T+1):
		x_t_ = f(x_t, np.zeros(2), 1)
		x_t = np.copy(x_t_)
		if np.linalg.norm(x_t,2) > 1e6: break
	return x_t

def finite_diff_grad(f, x0, eps):
    n = x0.shape[0]
    grad = np.zeros(n)

    for i in range(n):
        x_perturb = np.zeros(n)
        x_perturb[i] = eps/2
        grad[i] = (f(x0+x_perturb) - f(x0-x_perturb))/ (eps)
    return grad


def main():
	c1='y' #blue
	c2='red' #green

	#pi_nn, pi_nn_jax, jac_pi_nn = get_network_func()
	pi_nn = get_network_func_sac('QPendulum-v0', 'Gaussian', 0.2, 256)

	g = 10.
	m = 0.15
	l = 0.5
	mu = 0.05
	dt = 0.02
	T = 500

	def f_van(x,u, t):
		if t == 0:
			B = np.eye(2)
		else:
			B = np.zeros((2,2))
		x = x + B@u
		x1_next = x[0] + x[1]*0.01
		x2_next = x[1] + (-(1-x[0]**2)*x[1] - x[0])*0.01
		return np.array([x1_next, x2_next])

	def f_pen(x, u, t):
		if t == 0:
			B = np.eye(2)
		else:
			B = np.zeros((2,2))
		x = x + B@u
		u_pi = np.clip(pi_nn(x), -0.5, 0.5) 
		#if t==0:
		#	print('initial state in function: {}'.format(x))
		#	print('supposed initial state: {}'.format(u))
		x_next = np.array([x[0] + x[1]*dt, x[1] + (g/l*np.sin(x[0])-mu/(m*l**2)*x[1]+1/(m*l**2)*(u_pi))*dt])
		return x_next


	# xv, yv, roa = monte_carlo_ROA(f_pen, T=200, N=100, eps=1)

	# with open('pendulum_sac_roa.npy', 'wb') as f:
	#    	np.save(f, xv)
	#    	np.save(f, yv)
	#    	np.save(f, roa)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	with open('pendulum_sac_roa.npy', 'rb') as f:
	 	xv = np.load(f)
	 	yv = np.load(f)
	 	roa = np.load(f)

	ax.pcolor(xv,yv,roa)
	#plt.show()
	P = np.array([[1,0],[0,1]])
	P_inv = np.linalg.inv(P)
	N = 5
	alpha = T
	eps = 1e-1
	R_min = 0.0001
	R_max = 10.0
	R_feas = R_min
	R_i = (R_min + R_max)/2
	dyn_fn = f_pen

	def J_XT(x0):
		xT = single_run(dyn_fn, T, x0)
		return 1/2*np.linalg.norm(xT , 2)**2
	
	for j in range(10):
		x0 = np.random.uniform(-1,1, size=(2)).astype('float')
		#x0 = np.ones(2)
		x0 = R_i*x0/P_norm(x0,P)
		is_stable=True
		print('Trying R_i={}'.format(R_i))
		for i in range(N):
			grad = finite_diff_grad(J_XT, x0, eps)
			#print('gradient: ', grad)
			x0_next = -P_inv@grad
			#print(x0_next)
			x0_next = R_i*x0_next/P_norm(x0_next, P)
			x0 = np.copy(x0_next)
			xT = single_run(dyn_fn, T, x0)

			print("iteration: {}, x(0): {}, |x(T)|^2: {}".format(i, x0, np.linalg.norm(xT, 2)))
			if np.linalg.norm(xT, 2) > 1:
				is_stable = False
				break
		if is_stable:
			R_feas = R_i
			R_min = R_i
			x_bad_feas = x0
		else:
			R_max = R_i
			x_bad_max = x0
		R_i = (R_min + R_max)/2
		print('iter: ', j, 'Best feasible R:', R_feas, 'Outer R:', R_max)
		#print('worst-case initial condition: {}, norm: {}, stable: {}'.format(x0, np.linalg.norm(xT, 2), stable))


	elli_test = plot_ellipse(P, R_max)
	#circle = plot_circle( 1.18659970703125)
	ax.plot(elli_test[0,:], elli_test[1,:], c='c', label=r'ellipse Approximation')
	#ax.plot(circle[0,:], circle[1,:],c='g',label=r'spherical power-iteration estimate')
	ax.legend()
	plt.show()
if __name__ == "__main__":
	main()


