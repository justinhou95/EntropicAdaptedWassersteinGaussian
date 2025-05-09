import matplotlib.pyplot as plt

from cpl import bigau_cpl, gau_aw_cpl, gau_w_cpl, gau_eaw_cpl, gau_id_cpl, gau_ew_cpl
from plotting import (
    plot_gaussian_contor,
    plot_line_interpolation,
    plot_scatter_interpolation,
    plot_contor_interpolation,
)


def compare_rho(a, b, A, B):
    fig, ax = plt.subplots(1, 4, figsize=[16, 4], sharex=True, sharey=True)
    cmap = "plasma"

    for i, rho in enumerate([-1, 0, 1]):
        plt.sca(ax[i])
        plot_scatter_interpolation(a, b, A, B, gau_aw_cpl, rho=rho, cmap=cmap)
        plot_contor_interpolation(a, b, A, B, gau_aw_cpl, rho=rho)
        plt.title(rf"Adapted 2-Wasserstein ($\rho = {rho}$)")

    plt.sca(ax[3])
    plot_scatter_interpolation(a, b, A, B, gau_cpl=bigau_cpl, cmap=cmap)
    plot_gaussian_contor(a, A, "r")
    plot_gaussian_contor(b, B, "r")
    plt.title(r"Adapted 2-Wasserstein (non-Gaussian)")

    plt.savefig("eaw_opt_cpl.png", bbox_inches="tight")


def compare_scatter_contor(a, b, A, B):
    fig, ax = plt.subplots(1, 4, figsize=[16, 4], sharex=True, sharey=True)

    plt.sca(ax[0])
    plot_scatter_interpolation(a, b, A, B, gau_w_cpl)
    plot_contor_interpolation(a, b, A, B, gau_w_cpl)
    plt.title(r"2-Wasserstein")

    plt.sca(ax[1])
    plot_scatter_interpolation(a, b, A, B, gau_ew_cpl, lamb=1)
    plot_contor_interpolation(a, b, A, B, gau_ew_cpl, lamb=1)
    plt.title(r"Entropic 2-Wasserstein ($\lambda = 1$)")

    plt.sca(ax[2])
    plot_scatter_interpolation(a, b, A, B, gau_aw_cpl, rho=-1)
    plot_contor_interpolation(a, b, A, B, gau_aw_cpl, rho=-1)
    plt.title(r"Adapted 2-Wasserstein")

    plt.sca(ax[3])
    plot_scatter_interpolation(a, b, A, B, gau_eaw_cpl, lamb=0.5)
    plot_contor_interpolation(a, b, A, B, gau_eaw_cpl, lamb=0.5)
    plt.title(r"Entropic adapted 2-Wasserstein ($\lambda = 1$)")

    plt.savefig("scatter_contor.png", bbox_inches="tight")


def compare_scatter(a, b, A, B):
    fig, ax = plt.subplots(1, 4, figsize=[16, 4], sharex=True, sharey=True)

    plt.sca(ax[0])
    plot_scatter_interpolation(a, b, A, B, gau_w_cpl)
    plt.title(r"2-Wasserstein")

    plt.sca(ax[1])
    plot_scatter_interpolation(a, b, A, B, gau_ew_cpl, lamb=1)
    plt.title(r"Entropic 2-Wasserstein ($\lambda = 1$)")

    plt.sca(ax[2])
    plot_scatter_interpolation(a, b, A, B, gau_aw_cpl, rho=0)
    plt.title(r"Adapted 2-Wasserstein")

    plt.sca(ax[3])
    plot_scatter_interpolation(a, b, A, B, gau_eaw_cpl, lamb=1)
    plt.title(r"Entropic adapted 2-Wasserstein ($\lambda = 1$)")

    plt.savefig("scatter.png", bbox_inches="tight")


def compare_contor(a, b, A, B):
    fig, ax = plt.subplots(1, 4, figsize=[16, 4], sharex=True, sharey=True)

    plt.sca(ax[0])
    plot_contor_interpolation(a, b, A, B, gau_w_cpl)
    plt.title(r"2-Wasserstein")

    plt.sca(ax[1])
    plot_contor_interpolation(a, b, A, B, gau_ew_cpl, lamb=1)
    plt.title(r"Entropic 2-Wasserstein ($\lambda = 1$)")

    plt.sca(ax[2])
    plot_contor_interpolation(a, b, A, B, gau_aw_cpl)
    plt.title(r"Adapted 2-Wasserstein")

    plt.sca(ax[3])
    plot_contor_interpolation(a, b, A, B, gau_eaw_cpl, lamb=1)
    plt.title(r"Entropic adapted 2-Wasserstein ($\lambda = 1$)")

    plt.savefig("scatter.png", bbox_inches="tight")


def compare_line(a, b, A, B):
    fig, ax = plt.subplots(1, 4, figsize=[16, 4], sharex=True, sharey=True)

    plt.sca(ax[0])
    plot_line_interpolation(a, b, A, B, gau_w_cpl)
    plt.title(r"2-Wasserstein")

    plt.sca(ax[1])
    plot_line_interpolation(a, b, A, B, gau_ew_cpl, lamb=1)
    plt.title(r"Entropic 2-Wasserstein ($\lambda = 1$)")

    plt.sca(ax[2])
    plot_line_interpolation(a, b, A, B, gau_aw_cpl)
    plt.title(r"Adapted 2-Wasserstein")

    plt.sca(ax[3])
    plot_line_interpolation(a, b, A, B, gau_eaw_cpl, lamb=1)
    plt.title(r"Entropic adapted 2-Wasserstein ($\lambda = 1$)")

    plt.savefig("line.png", bbox_inches="tight")
