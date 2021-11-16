from hinf_syn import get_H_inf_norm
import numpy as np


A = np.array([[1.0, 0.1],[0.0, 1.0]])
B1 = np.array([[0.0],[0.1]])
K = np.array([[-2.0,-2.0]])
B2 = B1
C1 = np.array([[1.0,0],[0,1.0],[0,0]])
D11 = np.zeros((3,1))
D12 = np.array([[0],[0],[0.1]])

gamma = get_H_inf_norm(A, B1, B2, C1, D11, D12, K, D_l=None, D_r_inv=None)
print(gamma)