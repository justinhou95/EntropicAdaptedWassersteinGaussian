import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import ot.plot
import seaborn as sns
from scipy.stats import multivariate_normal

from cpl import bigau_cpl, gau_aw_cpl, gau_w_cpl, gau_eaw_cpl, gau_id_cpl


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


def plot_scatter_interpolation(a, b, A, B, gau_cpl, n_sample=1000, n_time=5, **kwargs):
    r""" 
    
    """
    cmap = mpl.colormaps[kwargs.get("cmap", "viridis")]
    # cmap = mpl.colormaps["plasma"]
    # cmap = mpl.colormaps["cividis"]
    if gau_cpl == bigau_cpl:
        X, Y = bigau_cpl(a, b, A, B)
    else:
        d = len(a)
        m, M = gau_cpl(a, b, A, B, **kwargs)
        XY = np.random.multivariate_normal(m, M, size=n_sample)
        X = XY[:, :d]
        Y = XY[:, d:]
    for t in np.linspace(0, 1, n_time):
        Xt = (1 - t) * X + t * Y
        # Xt = t*m + X.dot(((1-t)+t*M).T)
        c = np.zeros([len(Xt), 4]) + cmap(t)
        plt.scatter(Xt[:, 0], Xt[:, 1], s=10, c=c, alpha=0.3)
    plot_gaussian_contor(a, A, "r")
    plot_gaussian_contor(b, B, "r")
    return X, Y


def plot_contor_interpolation(a, b, A, B, gau_cpl, **kwargs):
    cmap = mpl.colormaps["viridis"]
    # cmap = mpl.colormaps["plasma"]
    # cmap = mpl.colormaps["cividis"]

    m, M = gau_cpl(a, b, A, B, **kwargs)
    I = np.identity(len(A))
    for i, t in enumerate(np.linspace(0, 1, 5)):
        tt = np.block([(1 - t) * I, t * I])
        mt = tt.dot(m)
        Mt = tt.dot(M).dot(tt.T)
        color = cmap(t)
        try:
            plot_gaussian_contor(mt, Mt, "r")
        except:
            print("Cant not plot covariance: \n", Mt)
            print(Mt)
            Mt = Mt + 1e-5 * I
            # print("Instead we plot for: \n", Mt)
            plot_gaussian_contor(mt, Mt, "r")


def plot_line_interpolation(a, b, A, B, gau_cpl, **kwargs):
    d = len(a)
    m, M = gau_cpl(a, b, A, B, **kwargs)
    XY = np.random.multivariate_normal(m, M, size=1000)
    X = XY[:, :d]
    Y = XY[:, d:]

    ns = 50
    Xs = X[:ns]
    Ys = Y[:ns]
    Gs = np.diag(np.ones(len(Xs)))

    ot.plot.plot2D_samples_mat(Xs, Ys, Gs, alpha=0.1, c="k")
    plt.plot(X[:, 0], X[:, 1], "+b", label="Source samples")
    plt.plot(Y[:, 0], Y[:, 1], "xg", label="Target samples")
    plt.legend()
    plot_gaussian_contor(a, A, "r")
    plot_gaussian_contor(b, B, "r")
