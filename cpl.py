import numpy as np
import scipy
import matplotlib.pyplot as plt
import warnings


# Sigma double used, we can use, mu Sigma
def sqrtm(X):
    r"""
    Square root of matrix
    """
    return np.array(scipy.linalg.sqrtm(X))

class Cpl:
    def __init__(self, a, b, A, B):
        self.a = a 
        self.b = b 
        self.A = A
        self.B = B
        self.d = len(a)

class GauCpl(Cpl):
    def __init__(self, a, b, A, B):
        super().__init__(a, b, A, B)
        self.mu_Sigma()

    def mu_Sigma(self):
        self.C = np.diag(np.zeros(len(self.A)))
        self.Sigma = np.block([[self.A, self.C], [self.C.T, self.B]])
        self.mu = np.block([self.a, self.b])
        return self.mu, self.Sigma
    
    def samples(self, n_sample):
        XY = np.random.multivariate_normal(self.mu, self.Sigma, size=n_sample)
        X = XY[:, :self.d]
        Y = XY[:, self.d:]
        return X,Y 
    
class 

    



def bigau_cpl(a, b, A, B, **kwargs):
    r"""
    Bi-gaussian optimal coupling (non-Gaussian)
    """
    L = np.linalg.cholesky(A)
    Sigma = np.linalg.cholesky(B)
    O = np.zeros_like(A)
    T = np.block([[L, O], [O, Sigma]])
    m = np.block([a, b])

    noiseX = np.random.normal(size=[1000, 2])
    noiseW = 2 * np.random.binomial(1, 0.5, size=1000) - 1
    noiseY = np.zeros_like(noiseX)
    noiseY[:, 0] += noiseX[:, 0] * noiseW
    noiseY[:, 1] += noiseX[:, 1]

    noiseXY = np.concatenate([noiseX, noiseY], axis=1)
    XY = T.dot(noiseXY.T).T + m
    X = XY[:, :2]
    Y = XY[:, 2:]
    return X, Y


def gau_id_cpl(a, b, A, B, **kwargs):
    r"""
    Independent coupling
    """
    C = np.diag(np.zeros(len(A)))
    Sigma = np.block([[A, C], [C.T, B]])
    m = np.block([a, b])
    return m, Sigma


def gau_w_cpl(a, b, A, B, **kwargs):
    r"""
    2-Wasserstein optimal coupling
    """
    Aroot = sqrtm(A)
    Arootinv = np.linalg.pinv(Aroot)
    I = np.identity(len(A))
    D = sqrtm(4 * Aroot.dot(B).dot(Aroot))
    C = 0.5 * Aroot.dot(D).dot(Arootinv)

    Sigma = np.block([[A, C], [C.T, B]])
    m = np.block([a, b])

    return m, Sigma


def gau_ew_cpl(a, b, A, B, lamb, **kwargs):
    r"""
    Entropic 2-Wasserstein optimal coupling
    """
    Aroot = sqrtm(A)
    Arootinv = np.linalg.pinv(Aroot)
    I = np.identity(len(A))
    D = sqrtm(4 * Aroot.dot(B).dot(Aroot) + lamb**2 / 4 * I)
    C = 0.5 * Aroot.dot(D).dot(Arootinv) - lamb / 4 * I

    Sigma = np.block([[A, C], [C.T, B]])
    m = np.block([a, b])

    return m, Sigma


def gau_aw_cpl(a, b, A, B, rho=0, **kwargs):
    r"""
    Adapted 2-Wasserstein optimal coupling
    """

    L = np.linalg.cholesky(A)
    Sigma = np.linalg.cholesky(B)

    diag_LTM = np.diag(L.T.dot(Sigma))
    diag_P = np.zeros_like(diag_LTM)
    diag_P[diag_LTM > 0] += 1
    diag_P[diag_LTM < 0] += -1
    diag_P[diag_LTM == 0] += rho
    if any(diag_LTM == 0):
        print(f"The optimizer is not unique, return the one with rho = {rho:.2f}")

    P = np.diag(diag_P)

    C = L.dot(P).dot(Sigma.T)
    Sigma = np.block([[A, C], [C.T, B]])
    m = np.block([a, b])

    return m, Sigma


def func_rho(x, lamb):
    rho = np.zeros_like(x)
    x1 = x[x != 0]
    rho1 = (np.sqrt(16 * x1**2 + lamb**2) - lamb) / 4 / x1
    rho[x != 0] = rho1
    return rho


def gau_eaw_cpl(a, b, A, B, lamb=0, **kwargs):
    r"""
    Entropic adapted 2-Wasserstein optimal coupling
    """
    lamb = kwargs.get("lamb", lamb)
    L = np.linalg.cholesky(A)
    Sigma = np.linalg.cholesky(B)

    diag_LTM = np.diag(L.T.dot(Sigma))
    rho = func_rho(diag_LTM, lamb)
    P = np.diag(rho)

    C = L.dot(P).dot(Sigma.T)
    Sigma = np.block([[A, C], [C.T, B]])
    m = np.block([a, b])

    return m, Sigma
