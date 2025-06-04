import multiprocessing
import os
import shutil
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, override

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from scipy.optimize import OptimizeResult, minimize
from sympy import prime

from lib.funs import Elliptic, OptFun
from lib.lincmaes import CMAVariation
from lib.util import CMAExperimentCallback, average_interpolated_values
from lib.wrapper import eswrapper

BOUNDS = 3
DIM = 10
RESULT_DIR = Path(f"results/bfgs_comparison/dim_{DIM}")
NUM_RUNS = 10


class StopBFGS(Exception): ...


@dataclass
class BestValueEvalCounterCallback(CMAExperimentCallback):
    funccalls: list[int] = field(default_factory=list)
    evaluations: list[float] = field(default_factory=list)
    best_evaluations: list[float] = field(default_factory=list)
    best: tuple[np.ndarray, float] | None = field(default=None)

    @override
    def __call__(self, es):
        self.funccalls.append(es.countevals)
        self.evaluations.append(es.best.f)

        if self.best is None or es.best.f < self.best[1]:
            self.best = (es.mean.copy(), es.best.f)
            logger.info(f"New best value: {self.best[1]} at {self.best[0]}")

        self.best_evaluations.append(self.best[1])


class EvalCounter(OptFun):
    """A wrapper to count the number of evaluations"""

    fun: Callable
    nfev: int
    best: tuple[np.ndarray, float] | None

    def __init__(self, fun: Callable):
        self.fun = fun
        self.nfev = 0
        self.best = None

    def __call__(self, x):
        self.nfev += 1
        y = self.fun(x)

        if self.best is None or y < self.best[1]:
            self.best = (x.copy(), y)

        return y


def single_comparison(i: int):
    """Run BGFS and CMA-ES with a randomized starting point and save
    their results to file"""
    seed: int = prime(i)  # pyright: ignore[reportAssignmentType]
    rng = np.random.default_rng(seed)
    x = (rng.random(DIM) - 0.5) * 2 * 5
    maxevals = 4000 * DIM

    bgfs_evals, bgfs_best = np.array([]), np.array([])
    elliptic_counter = EvalCounter(Elliptic.fun)

    def log_bgfs(intermediate_result: OptimizeResult):
        nonlocal bgfs_evals, bgfs_best, elliptic_counter
        logger.info(f"Thread {i}: {elliptic_counter.nfev} BFGS evals")

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
    callback = BestValueEvalCounterCallback()
    _ = eswrapper(
        x=x,
        fun=Elliptic,
        popsize=4 * DIM,
        maxevals=maxevals,
        variation=CMAVariation.VANILLA,
        seed=seed,
        callback=callback,
    )
    np.savetxt(
        RESULT_DIR / "cma" / f"{i}.csv",
        np.column_stack(  # pyright: ignore[reportCallIssue]
            (
                callback.funccalls,
                callback.best_evaluations,
            )
        ),
        delimiter=",",
        header="evals, best",
    )
    logger.info(f"{i}: done with CMA-ES")


def main():
    with Pool(multiprocessing.cpu_count()) as pool:
        pool.map(single_comparison, range(1, NUM_RUNS + 1))


def plot_results():
    bfgs_results = []
    for i in range(1, NUM_RUNS + 1):
        bfgs_results.append(np.loadtxt(RESULT_DIR / "bfgs" / f"{i}.csv", delimiter=","))

    bfgs_evals = [[row[0] for row in br] for br in bfgs_results]
    bfgs_values = [[row[1] for row in br] for br in bfgs_results]
    maxevals = max(max(evals) for evals in bfgs_evals)

    x_bfgs, bfgs_interpolated = average_interpolated_values(
        bfgs_values, bfgs_evals, maxevals
    )

    cma_results = []
    for i in range(1, NUM_RUNS + 1):
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
    plt.yscale("log")
    plt.savefig(RESULT_DIR / f"plot_{DIM}.png")


if __name__ == "__main__":
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR / "bfgs", exist_ok=True)
    os.makedirs(RESULT_DIR / "cma", exist_ok=True)

    main()
    plot_results()
