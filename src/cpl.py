import numpy as np
import scipy


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


class GauIdCpl(Cpl):
    r"""
    Independent coupling
    """

    def __init__(self, a, b, A, B):
        super().__init__(a, b, A, B)
        self.mu, self.Sigma = self.mu_Sigma()

    def mu_Sigma(self):
        C = np.diag(np.zeros(len(self.A)))
        Sigma = np.block([[self.A, C], [C.T, self.B]])
        mu = np.block([self.a, self.b])
        return mu, Sigma

    def samples(self, n_sample):
        XY = np.random.multivariate_normal(self.mu, self.Sigma, size=n_sample)
        X = XY[:, : self.d]
        Y = XY[:, self.d :]
        return X, Y


class GauWassCpl(GauIdCpl):
    r"""
    2-Wasserstein optimal coupling
    """

    def __init__(self, a, b, A, B):
        super().__init__(a, b, A, B)

    def mu_Sigma(self):
        Aroot = sqrtm(self.A)
        Arootinv = np.linalg.pinv(Aroot)
        I = np.identity(len(self.A))
        D = sqrtm(Aroot.dot(self.B).dot(Aroot))
        C = Aroot.dot(D).dot(Arootinv)
        Sigma = np.block([[self.A, C], [C.T, self.B]])
        mu = np.block([self.a, self.b])
        return mu, Sigma


class GauEntWassCpl(GauIdCpl):
    r"""
    Entropic 2-Wasserstein optimal coupling
    """

    def __init__(self, a, b, A, B, lamb):
        self.lamb = lamb
        super().__init__(a, b, A, B)

    def mu_Sigma(self):
        Aroot = sqrtm(self.A)
        Arootinv = np.linalg.pinv(Aroot)
        I = np.identity(len(self.A))
        D = sqrtm(4 * Aroot.dot(self.B).dot(Aroot) + self.lamb**2 / 4 * I)
        C = 0.5 * Aroot.dot(D).dot(Arootinv) - self.lamb / 4 * I
        Sigma = np.block([[self.A, C], [C.T, self.B]])
        mu = np.block([self.a, self.b])
        return mu, Sigma


class GauAdaWassCpl(GauIdCpl):
    r"""
    Adapted 2-Wasserstein optimal coupling
    """

    def __init__(self, a, b, A, B, rho=0):
        self.rho = rho
        super().__init__(a, b, A, B)

    def mu_Sigma(self):
        L = np.linalg.cholesky(self.A)
        M = np.linalg.cholesky(self.B)

        diag_LTM = np.diag(L.T.dot(M))
        diag_P = np.zeros_like(diag_LTM)
        diag_P[diag_LTM > 0] += 1
        diag_P[diag_LTM < 0] += -1
        if any(diag_LTM == 0):
            # The optimizer is not unique, return the one with rho
            diag_P[diag_LTM == 0] += self.rho
        P = np.diag(diag_P)
        C = L.dot(P).dot(M.T)

        Sigma = np.block([[self.A, C], [C.T, self.B]])
        mu = np.block([self.a, self.b])
        return mu, Sigma


class GauEntAdaWassCpl(GauIdCpl):
    r"""
    Entropic Adapted 2-Wasserstein optimal coupling
    """

    def __init__(self, a, b, A, B, lamb):
        self.lamb = lamb
        super().__init__(a, b, A, B)

    def _func_rho(self, x, lamb):
        rho = np.zeros_like(x)
        x1 = x[x != 0]
        rho1 = (np.sqrt(16 * x1**2 + lamb**2) - lamb) / 4 / x1
        rho[x != 0] = rho1
        return rho

    def mu_Sigma(self):
        # lamb = kwargs.get("lamb", self.lamb)
        L = np.linalg.cholesky(self.A)
        M = np.linalg.cholesky(self.B)

        diag_LTM = np.diag(L.T.dot(M))

        rho = self._func_rho(diag_LTM, self.lamb)
        P = np.diag(rho)
        C = L.dot(P).dot(M.T)

        Sigma = np.block([[self.A, C], [C.T, self.B]])
        mu = np.block([self.a, self.b])
        return mu, Sigma


class BiGauCpl(Cpl):
    r"""
    Bi-gaussian optimal coupling (non-Gaussian)
    """

    def __init__(self, a, b, A, B):
        super().__init__(a, b, A, B)

    def samples(self, n_sample):
        L = np.linalg.cholesky(self.A)
        M = np.linalg.cholesky(self.B)
        O = np.zeros_like(self.A)
        T = np.block([[L, O], [O, M]])
        m = np.block([self.a, self.b])

        noiseX = np.random.normal(size=[n_sample, 2])
        noiseW = 2 * np.random.binomial(1, 0.5, size=n_sample) - 1
        noiseY = np.zeros_like(noiseX)
        noiseY[:, 0] += noiseX[:, 0] * noiseW
        noiseY[:, 1] += noiseX[:, 1]

        noiseXY = np.concatenate([noiseX, noiseY], axis=1)
        XY = T.dot(noiseXY.T).T + m
        X = XY[:, :2]
        Y = XY[:, 2:]
        return X, Y
