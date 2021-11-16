from cvxpy import Variable, Problem, Minimize, bmat, SCS, MOSEK
import cvxpy as cp
import numpy as np
#import gurobipy
import mosek
import matlab.engine
#eng = matlab.engine.start_matlab()


def get_robust_stabilizing_K(A, B1, B2, C1, D11, D12, D_l=None, D_r_inv=None):
    (nx, nu) = B2.shape
    (_ , nd) = B1.shape
    (nz, _) = C1.shape
    if D_l is not None and D_r_inv is not None:
        B1 = B1 @ D_r_inv
        C1 = D_l @ C1
        D11 = D_l @ D11 @ D_r_inv
        D12 = D_l @ D12
    Q = Variable((nx, nx), symmetric=True)
    L = Variable((nu, nx))
    eta = Variable()
    Id = np.eye(nd)
    Iz = np.eye(nz)
    Onxz = np.zeros((nx, nz))
    Onxd = np.zeros((nx, nd))
    Ondz = np.zeros((nd, nz))

    #M11 = -Q + B1 @ B1.T
    #M12 = A @ Q + B2 @ L
    #M13 = Onxz
    #M22 = -Q
    #M23 = (C1 @ Q + D2 @ L).T
    #M33 = -eta * Iz
    #LMI = bmat([[M11, M12, M13],
    #            [M12.T, M22, M23],
    #            [M13.T, M23.T, M33]])

    M11 = Q
    M12 = A @ Q + B2 @ L
    M13 = B1
    M14 = Onxz
    M22 = Q
    M23 = Onxd
    M24 = Q @ C1.T + L.T @ D12.T
    M33 = eta * Id
    M34 = D11.T#Ondz
    M44 = eta * Iz
    LMI = bmat([[M11, M12, M13, M14],
                [M12.T, M22, M23, M24],
                [M13.T, M23.T, M33, M34],
                [M14.T, M24.T, M34.T, M44]])

    #print('(nx, nu, nd, nz): ', nx, nu, nd, nz)
    constraints = []
    #constraints.append(Q.T == Q)
    constraints.append(Q >> 0)
    constraints.append(LMI >> 0)#*np.eye(2*nx + nd + nz))
    constraints.append(eta >= 0)
    obj = Minimize(eta)
    prob = Problem(obj, constraints)
    try:
        prob.solve(solver=MOSEK)
    except:
        prob.solve(solver=SCS, acceleration_lookback=0, max_iters=10000)

    gamma_ = eta.value
    Q_ = np.asmatrix(Q.value, dtype='float')
    Q_ = (Q_.T + Q_)/2
    w, v = np.linalg.eigh(Q_)
    w[w < 0] = 0
    Q_ = v @ np.diag(w) @ v.T
    L_ = np.asmatrix(L.value, dtype='float')
    P = np.linalg.inv(Q_)
    K = L_ * P
    return L_, Q_, K, gamma_

def get_H_inf_norm(A, B1, B2, C1, D11, D12, K, D_l=None, D_r_inv=None):
    #print('A {}, B1 {}, B2 {}'.format(A.shape, B1.shape, B2.shape))
    #print('C1 {},  D11 {}, D12 {}'.format(C1.shape, D11.shape, D12.shape))
    (nx, nd) = B1.shape
    (_, nu) = B2.shape
    (nz, _) = C1.shape

    if D_l is not None and D_r_inv is not None:
        B1 = B1 @ D_r_inv
        C1 = D_l @ C1
        D11 = D_l @ D11 @ D_r_inv
        D12 = D_l @ D12

    Ad = A + B2 @ K
    Bd = B1
    Dd = D11
    Cd = (C1 + D12 @ K)
    P = Variable((nx, nx), symmetric=True)
    gamma = Variable()
    Ind = np.eye(nd)
    Inz = np.eye(nz)
    Onxnz = np.zeros((nx, nz))
    Onxnd = np.zeros((nx, nd))
    Ondnz = np.zeros((nd, nz))


    # M11 = Ad.T@P@Ad - P
    # M12 = Ad.T@P@Bd
    # M13 = Cd.T
    # M22 = Bd.T@P@Bd - gamma * Ind
    # M23 = Dd.T
    # M33 = - gamma * Inz

    # LMI = bmat([    [M11, M12, M13],
    #                 [M12.T, M22, M23],
    #                 [M13.T, M23.T, M33]])
    M11 = P
    M12 = Ad@P
    M13 = Bd
    M14 = Onxnz
    M22 = P
    M23 = Onxnd
    M24 = P@Cd.T
    M33 = gamma*Ind
    M34 = Dd.T
    M44 = gamma * Inz

    LMI = bmat([ [M11, M12, M13, M14],
                   [M12.T, M22, M23, M24],
                   [M13.T, M23.T, M33, M34],
                   [M14.T, M24.T, M34.T, M44]])


    # M11 = P
    # M12 = P @ Ad
    # M13 = P @ Bd
    # M14 = Onxnz
    # M22 = P
    # M23 = Onxnd
    # M24 = Cd.T
    # M33 = gamma * Ind
    # M34 = Dd.T
    # M44 = gamma * Inz

    # LMI = bmat([ [M11, M12, M13, M14],
    #                [M12.T, M22, M23, M24],
    #                [M13.T, M23.T, M33, M34],
    #                [M14.T, M24.T, M34.T, M44]])
    constraints = []
    constraints.append(P >> 0.0)#np.eye(nx))
    constraints.append(LMI >> 0.0)
    constraints.append(gamma >= 0.0)
    obj = Minimize(gamma)
    prob = Problem(obj, constraints)
    try:
        prob.solve(solver=MOSEK)
    except:
        prob.solve(solver=SCS, acceleration_lookback=0, max_iters=10000)

    #prob.solve()
    gamma_ = gamma.value

    return gamma_

def get_H_inf_norm_matlab(A, B1, B2, C1, D11, D12, K, ts=0.1, D_l=None, D_r_inv=None):
    #print('A {}, B1 {}, B2 {}'.format(A.shape, B1.shape, B2.shape))
    #print('C1 {},  D11 {}, D12 {}'.format(C1.shape, D11.shape, D12.shape))
    (nx, nd) = B1.shape
    (_, nu) = B2.shape
    (nz, _) = C1.shape

    if D_l is not None and D_r_inv is not None:
        B1 = B1 @ D_r_inv
        C1 = D_l @ C1
        D11 = D_l @ D11 @ D_r_inv
        D12 = D_l @ D12

    Ad = A + B2 @ K
    Bd = B1
    Dd = D11
    Cd = (C1 + D12 @ K)
    
    Adm = matlab.double(Ad.tolist())
    Bdm = matlab.double(Bd.tolist())
    Ddm = matlab.double(Dd.tolist())
    Cdm = matlab.double(Cd.tolist())

    sys = eng.ss(Adm, Bdm, Cdm, Ddm, ts)
    gamma = eng.hinfnorm(sys)

    return gamma

def get_Hinffi_matlab(A, B1, B2, C1, D11, D12, ts=0.1):
    Ad = A
    Bd = np.block([B1, B2])
    Cd = C1
    Dd = np.block([D11,D12])
    
    Adm = matlab.double(Ad.tolist())
    Bdm = matlab.double(Bd.tolist())
    Ddm = matlab.double(Dd.tolist())
    Cdm = matlab.double(Cd.tolist())
    sys = eng.ss(Adm, Bdm, Cdm, Ddm, ts)
    K,CL,gamma = eng.hinffi(sys,1,nargout=3)
    print('hello agian' ,gamma)
    return np.asarray(K), gamma


