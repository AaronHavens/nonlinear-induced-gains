import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse(P, radius):
	R = np.linalg.inv(np.linalg.cholesky(P))
	t = np.linspace(0, 2*np.pi, 100) # or any high number to make curve smooth
	x = np.cos(t)*radius
	y = np.sin(t)*radius
	z = np.vstack([x, y])

	ellipse = (z.T@R).T;
		
	z_i = ellipse[:,10].reshape(2,1)
	#print(radius)
	#print('point',np.sqrt(z_i.T@P@z_i))
	return ellipse

def plot_circle(radius):
	t = np.linspace(0, 2*np.pi, 100) # or any high number to make curve smooth
	x = np.cos(t)*radius
	y = np.sin(t)*radius
	z = np.vstack([x, y])
	return z

def get_random_ellipsoid_a():
	a = np.random.uniform(1e-6,1)
	b = np.random.uniform(1e-6,1)
	theta = np.random.uniform(0,2*np.pi)
	R = np.array([[np.cos(theta),-np.sin(theta)],
				[np.sin(theta), np.cos(theta)]])
	P = np.array([[a,0],[0,b]])
	return R@P@R.T

def get_random_ellipsoid_b():
	A = np.random.normal(0,1,size=(2,2))
	A = 1/2*(A + A.T)
	return A@A.T

def ellipse_vol(P,r):
	return (np.linalg.det(P/r**2))**(-1/2)




