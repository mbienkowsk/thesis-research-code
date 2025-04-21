from multiprocessing import Pool
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path
from typing import Callable
from .comparison import average_interpolated_values
from lib.funs import Elliptic, OptFun
import numpy as np
from sympy import prime
from scipy.optimize import OptimizeResult, minimize

from lib.lincmaes import CMAVariation
from lib.wrapper import eswrapper

rng = np.random.default_rng(0)
BOUNDS = 3
DIM = 3
RESULT_DIR = Path("csv_results")


class StopBFGS(Exception): ...


class EvalCounter(OptFun):
    """A wrapper to count the number of evaluations"""

    fun: Callable
    nfev: int

    def __init__(self, fun: Callable):
        self.fun = fun
        self.nfev = 0

    def __call__(self, x):
        self.nfev += 1
        return self.fun(x)


def single_comparison(i: int):
    """Run BGFS and CMA-ES with a randomized starting point and save
    their results to file"""
    seed: int = prime(i)  # pyright: ignore[reportAssignmentType]
    x = (rng.random(DIM) - 0.5) * 2 * 5
    maxevals = 4000 * DIM

    bgfs_evals, bgfs_best = np.array([]), np.array([])
    cma_evals, cma_best = np.array([]), np.array([])
    elliptic_counter = EvalCounter(Elliptic.fun)

    def log_bgfs(intermediate_result: OptimizeResult):
        nonlocal bgfs_evals, bgfs_best, elliptic_counter

        if elliptic_counter.nfev > maxevals:
            raise StopBFGS()
        bgfs_evals = np.append(bgfs_evals, elliptic_counter.nfev)
        bgfs_best = np.append(bgfs_best, intermediate_result.fun)

    # BFGS
    try:
        minimize(
            elliptic_counter,
            x.copy(),
            method="BFGS",
            callback=log_bgfs,
        )
    finally:
        np.savetxt(
            RESULT_DIR / "bfgs" / f"{i}.csv",
            np.column_stack((bgfs_evals, bgfs_best)),
            delimiter=",",
            header="evals, best",
        )
    logger.info(f"{i}: done with BFGS")

    # CMA-ES
    cma_result = eswrapper(
        x=x,
        fun=Elliptic,
        popsize=4 * DIM,
        maxevals=maxevals,
        variation=CMAVariation.VANILLA,
        seed=seed,
    )
    np.savetxt(
        RESULT_DIR / "cma" / f"{i}.csv",
        np.column_stack((cma_result.nums_evals, cma_result.best_values)),
        delimiter=",",
        header="evals, best",
    )
    logger.info(f"{i}: done with CMA-ES")


def main():
    with Pool(multiprocessing.cpu_count()) as pool:
        pool.map(single_comparison, range(1, 52))


def plot_results():
    # BFGS results
    bfgs_results = []
    for i in range(1, 52):
        bfgs_results.append(np.loadtxt(RESULT_DIR / "bfgs" / f"{i}.csv", delimiter=","))

    bfgs_evals = [[row[0] for row in br] for br in bfgs_results]
    bfgs_values = [[row[1] for row in br] for br in bfgs_results]
    maxevals = max(max(evals) for evals in bfgs_evals)

    x_bfgs, bfgs_interpolated = average_interpolated_values(
        bfgs_values, bfgs_evals, maxevals
    )

    # CMA-ES results
    cma_results = []
    for i in range(1, 52):
        cma_results.append(np.loadtxt(RESULT_DIR / "cma" / f"{i}.csv", delimiter=","))

    cma_evals = [[row[0] for row in br] for br in cma_results]
    cma_values = [[row[1] for row in br] for br in cma_results]
    maxevals = max(max(evals) for evals in cma_evals)
    x_cma, cma_interpolated = average_interpolated_values(
        cma_values, cma_evals, maxevals
    )
    CUTOFF = 200

    fig = plt.figure()
    sns.lineplot(
        x=x_bfgs[:CUTOFF],
        y=bfgs_interpolated[:CUTOFF],
        label="BFGS",
        ax=fig.gca(),
    )
    sns.lineplot(
        x=x_cma[:CUTOFF],
        y=cma_interpolated[:CUTOFF],
        label="CMA-ES",
        ax=fig.gca(),
    )
    plt.title(f"BFGS vs CMA-ES - f. pokrzywiona w {DIM} wymiarach")
    plt.xlabel("Liczba ewaluacji f. celu")
    plt.ylabel("Najlepsze znalezione rozwiązanie")
    plt.savefig(RESULT_DIR / "comparison.png")


if __name__ == "__main__":
    main()
    plot_results()
