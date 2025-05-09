import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ot.plot
from scipy.stats import multivariate_normal
from typing import Union
from .cpl import GauIdCpl, BiGauCpl


def plot_gaussian_contor(mu, Sigma, color="k"):
    r"""
    Plot contor of gaussian distribution
    """
    N = 200
    X = np.linspace(-5, 10, N)
    Y = np.linspace(-10, 5, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, Sigma)
    Z = rv.pdf(pos)
    plt.contour(X, Y, Z, levels=[0.03], linewidths=0.6, colors=[color])


def plot_scatter_interpolation(
    cpl: Union[GauIdCpl, BiGauCpl], n_sample=1000, n_time=5, **kwargs
):
    r"""
    Plot the McCann displacement interpolation with scatter plots
    """
    cmap = mpl.colormaps[kwargs.get("cmap", "viridis")]

    X, Y = cpl.samples(n_sample)

    for t in np.linspace(0, 1, n_time):
        Xt = (1 - t) * X + t * Y
        c = np.zeros([len(Xt), 4]) + cmap(t)
        plt.scatter(Xt[:, 0], Xt[:, 1], s=10, c=c, alpha=0.3)

    plot_gaussian_contor(cpl.a, cpl.A, "r")
    plot_gaussian_contor(cpl.b, cpl.B, "r")

    return X, Y


def plot_contor_interpolation(cpl: GauIdCpl, **kwargs):

    m, M = cpl.mu_Sigma()
    I = np.identity(len(cpl.A))
    for i, t in enumerate(np.linspace(0, 1, 5)):
        t_vec = np.block([(1 - t) * I, t * I])
        mt = t_vec.dot(m)
        Mt = t_vec.dot(M).dot(t_vec.T)
        # c = cmap(t)
        try:
            plot_gaussian_contor(mt, Mt, "r")
        except:
            # Cant not plot covariance, instead we plot Mt+epsilon I
            Mt = Mt + 1e-5 * I
            plot_gaussian_contor(mt, Mt, "r")


def plot_line_interpolation(cpl: Union[GauIdCpl, BiGauCpl], n_sample=50, **kwargs):
    X, Y = cpl.samples(n_sample)
    Xs = X[:n_sample]
    Ys = Y[:n_sample]
    Gs = np.diag(np.ones(len(Xs)))

    ot.plot.plot2D_samples_mat(Xs, Ys, Gs, alpha=0.1, c="k")
    plt.plot(X[:, 0], X[:, 1], "+b", label="Source samples")
    plt.plot(Y[:, 0], Y[:, 1], "xg", label="Target samples")
    plt.legend()
    plot_gaussian_contor(cpl.a, cpl.A, "r")
    plot_gaussian_contor(cpl.b, cpl.B, "r")
