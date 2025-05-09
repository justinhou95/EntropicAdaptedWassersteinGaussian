import matplotlib.pyplot as plt
import numpy as np
from .cpl import (
    BiGauCpl,
    GauAdaWassCpl,
    GauEntAdaWassCpl,
    GauEntWassCpl,
    GauIdCpl,
    GauWassCpl,
    Cpl,
)
from .plotting import (
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
        cpl = GauAdaWassCpl(a, b, A, B, rho)
        plot_scatter_interpolation(cpl, cmap=cmap)
        plot_contor_interpolation(cpl, rho=rho)
        plt.title(rf"Adapted 2-Wasserstein ($\rho = {rho}$)")

    plt.sca(ax[3])
    cpl = BiGauCpl(a, b, A, B)
    plot_scatter_interpolation(cpl, cmap=cmap)
    plot_gaussian_contor(a, A, "r")
    plot_gaussian_contor(b, B, "r")
    plt.title(r"Adapted 2-Wasserstein (non-Gaussian)")


def compare_scatter_contor(a, b, A, B):
    fig, ax = plt.subplots(1, 4, figsize=[16, 4], sharex=True, sharey=True)

    cpls = [
        GauWassCpl(a, b, A, B),
        GauEntWassCpl(a, b, A, B, lamb=1),
        GauAdaWassCpl(a, b, A, B, rho=-1),
        GauEntAdaWassCpl(a, b, A, B, lamb=1),
    ]
    titles = [
        r"2-Wasserstein",
        r"Entropic 2-Wasserstein ($\lambda = 1$)",
        r"Adapted 2-Wasserstein",
        r"Entropic adapted 2-Wasserstein ($\lambda = 1$)",
    ]

    for i in range(4):
        plt.sca(ax[i])
        plot_scatter_interpolation(cpls[i])
        plot_contor_interpolation(cpls[i])
        plt.title(titles[i])


def compare_line(a, b, A, B):
    fig, ax = plt.subplots(1, 4, figsize=[16, 4], sharex=True, sharey=True)

    cpls = [
        GauWassCpl(a, b, A, B),
        GauEntWassCpl(a, b, A, B, lamb=1),
        GauAdaWassCpl(a, b, A, B, rho=-1),
        GauEntAdaWassCpl(a, b, A, B, lamb=1),
    ]
    titles = [
        r"2-Wasserstein",
        r"Entropic 2-Wasserstein ($\lambda = 1$)",
        r"Adapted 2-Wasserstein",
        r"Entropic adapted 2-Wasserstein ($\lambda = 1$)",
    ]

    for i in range(4):
        plt.sca(ax[i])
        plot_line_interpolation(cpls[i])
        plt.title(titles[i])
