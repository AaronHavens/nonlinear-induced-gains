import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from power_iteration_roa import *
from tqdm import tqdm
from nn_utils import get_network_func
from matplotlib.patches import Ellipse
from ellipsoid_utils import *
import time
plt.rcParams['text.usetex'] = True


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
def single_run(f, T, x0):
	x_t = np.copy(x0)
	for i in range(1,T+1):
		x_t_ = f(x_t, np.zeros(2), 1)
		x_t = np.copy(x_t_)
	return x_t

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

def main():
	c1='y' #blue
	c2='red' #green

	pi_nn, pi_nn_jax, jac_pi_nn = get_network_func()

	g = 10.
	m = 0.15
	l = 0.5
	mu = 0.05
	dt = 0.02


	def f_pen(x, u, t):
		if t == 0:
			B = np.eye(2)
		else:
			B = np.zeros((2,2))
		x = x + B@u
		u_pi = np.clip(pi_nn(x), -0.7, 0.7) 
		#if t==0:
		#	print('initial state in function: {}'.format(x))
		#	print('supposed initial state: {}'.format(u))
		x_next = np.array([x[0] + x[1]*dt, x[1] + (g/l*np.sin(x[0])-mu/(m*l**2)*x[1]+1/(m*l**2)*(u_pi))*dt])
		return x_next
	
	def f_pen_jax(x, u, t):
		# min(0.7 ,max(-0.7, pi(x)))
		if t == 0:
			B = jnp.eye(2)
		else:
			B = jnp.zeros((2,2))
		x = x + B@u
		u_pi = jnp.clip(pi_nn_jax(x), -0.7, 0.7)
		x_next = jnp.array([x[0] + x[1]*dt, x[1] + (g/l*jnp.sin(x[0])-mu/(m*l**2)*x[1]+1/(m*l**2)*(u_pi))*dt])
		return x_next
# 
	def y_pen(x):
		return np.array([x[0],x[1]])

	def y_pen_jax(x):
		return jnp.array([x[0],x[1]])

	def f_van(x, u, t):
		if t == 0:
			B = np.eye(2)
		else:
			B = np.zeros((2,2))
		x = x + B@u
		x1_next = x[0] + x[1]*0.01
		x2_next = x[1] + (-(1-x[0]**2)*x[1] - x[0])*0.01
		return np.array([x1_next, x2_next])
	
	def f_van_jax(x, u, t):
		if t == 0:
			B = jnp.eye(2)
		else:
			B = jnp.zeros((2,2))
		x = x + B@u
		x1_next = x[0] + x[1]*0.01
		x2_next = x[1] + (-(1-x[0]**2)*x[1] - x[0])*0.01
		return jnp.array([x1_next, x2_next])
	
	def is_roa(fn, fn_jax, yfn, yfn_jax, R_, P, T=500, iterations=2, eps=1e-1):
		R = R_#/np.sqrt(T)
		x0 = np.array([0.0, 0.0])
		u = np.zeros((T,2))
		
		u[0,:] = np.random.uniform(-1,1,size=(2,))
		#u[0,:] = np.ones(2)
		#t = np.linspace(0,10,N)
		u = u*R/L2_norm(u, P)
		
		for i in range(iterations):
			x, y = phi(fn, yfn, u, T, x0)

			x, u, y, nu = pi(x,u,y)
	
			gamma = psi(fn_jax, yfn_jax, nu, x, u, y, T, P)
			print('R: {}, gamma: {}'.format(R, gamma[0]))
			#print(gamma)
			u = theta(gamma, R, P)
			print('theta normalized',L2_norm(u,P))
			#print(u)
			#plt.plot(u,c=colorFader(c1,c2,(i+1)/iterations))
			x, y = phi(fn, yfn, u, T, x0)
			x_T = np.linalg.norm(x[-1],2)
			print('Worst-case State Norm: {}, state: {}'.format(x_T, u[0]))
			#print(u[0], L2_norm(u,P)*np.sqrt(T))

			if x_T > eps or np.isnan(x_T) :
				break
			#print(L2_norm(u,P))
		is_stable =  x_T <= eps
		
		return is_stable, x[0], x_T
	
	def find_R(f,f_jax,y, y_jax, P, iterations=10, T=500, eps=1e-1):
		R_min = 0.0001
		R_max = 10.0
		R_feas = R_min
		R_i = (R_min + R_max)/2
		for i in range(10):
			print('Trying R={}'.format(R_i))
			is_stable, x_bad, x_T = is_roa(f, f_jax, y, y_jax, R_i, P, T=T, eps=eps)	
			#print('is stable: {}, |X(T)|={}'.format(is_stable, x_T))			
			if is_stable:
				R_feas = R_i
				R_min = R_i
				x_bad_feas = x_bad
			else:
				R_max = R_i
				x_bad_max = x_bad
			R_i = (R_min + R_max)/2
			print('iter: ', i, 'Best feasible R:', R_feas, 'Outer R:', R_max)
		return R_max, x_bad

	#single_run(f_van, 5000, np.array([-1.85827186,  0.35427496]))
	P_test = weighting_P(0/360*2*np.pi, 10,1)
	print(ellipse_vol(P_test, 3.1))
	P_heyin = np.array([[2.7358, 0.5736],[0.5736, 1.4734]])
	print(ellipse_vol(P_heyin,1))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#xv, yv, roa = monte_carlo_ROA(f_pen, T=500, N=100, eps=1e-1)

	# with open('pendulum_roa.npy', 'wb') as f:
	#  	np.save(f, xv)
	#  	np.save(f, yv)
	#  	np.save(f, roa)


	with open('pendulum_roa.npy', 'rb') as f:
	 	xv = np.load(f)
	 	yv = np.load(f)
	 	roa = np.load(f)
	ax.pcolor(xv,yv,roa)
	#plt.show()
	#R_0, x_bad = find_R(f_pen, f_pen_jax, y_pen, y_pen_jax, P_test, T=500, eps=1e-1, iterations=15)
	R_0 = 2.9479255171775813
	x_bad = np.array([0.7478627,  1.75990869])
	x_bad1 = np.array([-0.74684914, -1.76431914])
	N_samples = 10
	best_R = 1.0
	best_vol = 0.0
	#print(x_bad)
	elli_test = plot_ellipse(P_test, R_0)
	ax.scatter(x_bad[0],x_bad[1],c='r')
	ax.scatter(x_bad1[0], x_bad1[1], c='r')
	elli_heyin = plot_ellipse(P_heyin, 1.0)
	#circle = plot_circle( 1.18659970703125)
	ax.plot(elli_test[0,:], elli_test[1,:], c='c', label=r'ellipse Approximation')
	ax.plot(elli_heyin[0,:], elli_heyin[1,:], c='b', label=r'Galaxys LMI Inner Approximation')
	#ax.plot(circle[0,:], circle[1,:],c='g',label=r'spherical power-iteration estimate')
	ax.legend()
	plt.show()
	for i in range(N_samples):
		P_i = get_random_ellipsoid_a()
		R_i, x_bad_i = find_R(f_pen, f_pen_jax, y_pen, y_pen_jax, P_i,T=500)
		vol_i = ellipse_vol(P_i, R_i)
		if vol_i > best_vol:
			best_vol = vol_i
			best_R = R_i
			best_P = P_i
		elli = plot_ellipse(P_i, R_i)
		ax.plot(elli[0,:], elli[1,:], label='sample {}'.format(i))
		ax.scatter([x_bad_i[0],-x_bad_i[0]],[x_bad_i[1],-x_bad_i[1]], c='c')
		ax.legend()
		fig.savefig('output.png')
		#ax.plot(elli_heyin[0,:], elli_heyin[1,:], c='b', label=r'Galaxys LMI Inner Approximation')
		#ax.plot(circle[0,:], circle[1,:],c='g',label=r'spherical power-iteration estimate')
		#plt.legend()
		#plt.show()
		print('outer iter: {}, this volume: {}, best volume: {}'.format(i, vol_i, best_vol))
		print('P_i:', P_i)
	
	
	elli = plot_ellipse(best_P, best_R)
	ax.plot(elli[0,:], elli[1,:],c='r', linestyle='*',linewidth=2, label='best sample')
	#ax.plot(elli_heyin[0,:], elli_heyin[1,:], c='b', label=r'Galaxys LMI Inner Approximation')
	#ax.plot(circle[0,:], circle[1,:],c='g',label=r'spherical power-iteration estimate')
	ax.axis('equal')
	ax.set_xlabel(r'$$\theta$$')
	ax.set_ylabel(r'$$\dot \theta$$')
	ax.set_xlim(-3.0,3.0)
	ax.set_ylim(-3.0,3.0)
	ax.legend()
	plt.show()


	
if __name__ == "__main__":
	main()