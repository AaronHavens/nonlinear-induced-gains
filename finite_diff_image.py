from power_iteration_roa import *
import numpy as np
from nn_utils import get_network_func, get_network_func_sac
from tqdm import tqdm
from ellipsoid_utils import *
from stable_baselines3 import SAC
import gym
import gym_custom
from matplotlib.pyplot import cm

def simulate_policy(env, model, x0, T=200):
	env.reset()
	obs = env.set_state(x0)
	#print(env.env.state, obs[-1])
	for i in range(T):
		action, _states = model.predict(obs, deterministic=True)
		obs,r,done,_ = env.step(action)
	env.close()
	return env.env.state

def monte_carlo_ROA(env, model, T, N, eps=1e-3):
	x = np.linspace(-2.0, 2.0, N)
	y = np.linspace(-2.0, 2.0, N)
	xv, yv = np.meshgrid(x, y)
	ROA = np.zeros((N,N))
	pbar = tqdm(total=int(N*N))
	for i in range(N):
		for j in range(N):
			x_t = np.array([xv[i,j], yv[i,j]])
			#print(x_t)
			x_T = simulate_policy(env, model, x_t, T=T)
			x_T_norm = np.linalg.norm(x_T,2)
			#print(x_t_norm)
			is_stable = x_T_norm <= eps
			ROA[i,j] = is_stable
			pbar.update(1)
		
	#out = np.column_stack((xv.ravel(), yv.ravel(), ROA.ravel()))
	#return out

	return xv, yv, ROA

def contour_plot(env, model, N):
	x = np.linspace(-2.0, 2.0, N)
	y = np.linspace(-2.0, 2.0, N)
	xv, yv = np.meshgrid(x, y)
	vx = np.zeros((N,N))
	vy = np.zeros((N,N))
	MAG = np.zeros((N,N))
	ROA = np.zeros((N,N))
	pbar = tqdm(total=int(N*N))
	env.reset()
	for i in range(N):
		for j in range(N):
			x_t = np.array([xv[i,j], yv[i,j]])
			obs = env.set_state(x_t)
			action, _states = model.predict(obs, deterministic=True)
			V = env.get_velocity(x_t, action)
			norm = np.linalg.norm(V, 2)
			vx[i,j] = V[0]/norm
			vy[i,j] = V[1]/norm
			print(V)
			MAG[i,j] = norm
			pbar.update(1)
	return xv, yv, vx, vy, MAG


	return 


def P_norm(x_, P):
	x = x_.reshape(x_.shape[0],1)
	Pnorm = np.sqrt(x.T@P@x)
	return Pnorm[0,0]

def finite_diff_grad(f, x0, eps):
    n = x0.shape[0]
    grad = np.zeros(n)

    for i in range(n):
        x_perturb = np.zeros(n)
        x_perturb[i] = eps/2
        grad[i] = (f(x0+x_perturb) - f(x0-x_perturb))/ (eps)
    return grad


def main():

	env = gym.make("ImPendulum-v0")
	model = SAC.load("train_RL/im_pen_model_11000_steps")
	T = 150
	eps=1.0

	# xv, yv, roa = monte_carlo_ROA(env, model, T=150, N=100, eps=1)

	# with open('pendulum_im_sac_roa.npy', 'wb') as f:
	#     np.save(f, xv)
	#     np.save(f, yv)
	#     np.save(f, roa)
	xv, yv, vx, vy, MAG = contour_plot(env, model, N=40)
	#cp = plt.contourf(xv,yv, MAG)
	#cb = plt.colorbar(cp)
	#quiv = plt.quiver(xv,yv, vx, vy,color='red',headlength=10)
	plt.streamplot(xv,yv,vx,vy, density=1.4)
	elli_test = plot_ellipse(np.eye(2), .71)
	#circle = plot_circle( 1.18659970703125)
	plt.plot(elli_test[0,:], elli_test[1,:], c='c', label=r'ellipse Approximation')
	plt.show()
	def J_XT(x0):
		xT = simulate_policy(env, model, x0, T=T)
		return 1/2*np.linalg.norm(xT , 2)**2

	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	with open('pendulum_im_sac_roa.npy', 'rb') as f:
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
			xT = simulate_policy(env, model, x0, T=T)

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
