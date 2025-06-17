"""Since es.C approximates the inverse of the hessian matrix, test how plugging it into
a gradient method such as BFGS compares to the vanilla CMA-ES and vanilla BFGS/L-BFGS"""

import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cma.evolution_strategy import CMAEvolutionStrategy
from loguru import logger
from scipy.optimize import bracket, golden, minimize
from sympy import prime

from lib.funs import Elliptic
from lib.lincmaes import CMAVariation
from lib.util import (BestValueEvalCounterCallback,
                      BFGSBestValueEvalCounterCallback, CMAExperimentCallback,
                      EvalCounter, gradient_central,
                      load_and_interpolate_results)
from lib.wrapper import eswrapper

BOUNDS = 100
DIM = 10
MAXEVALS = 4000 * DIM
K_VALUE = 1
SWITCH_AFTER = K_VALUE * DIM
RESULT_DIR = Path(f"results/dim_{DIM}/K_{K_VALUE}")
NUM_RUNS = 25
VANILLA_RESULT_DIR = RESULT_DIR / "vanilla"
BFGS_RESULT_DIR = RESULT_DIR / "bfgs"
LBFGS_RESULT_DIR = RESULT_DIR / "lbfgs"
LINESEARCH_RESULT_DIR = RESULT_DIR / "linesearch"
CMABFGS_RESULT_DIR = RESULT_DIR / "cmabfgs"

PLOT_X_CUTOFF = 200


def run_vanilla(x: np.ndarray, seed: int, idx: int):
    callback = BestValueEvalCounterCallback()
    _ = eswrapper(
        x=x,
        fun=Elliptic,
        popsize=4 * DIM,
        maxevals=MAXEVALS,
        variation=CMAVariation.VANILLA,
        seed=seed,
        callback=callback,
    )
    np.savetxt(
        VANILLA_RESULT_DIR / f"{idx}.csv",
        np.column_stack(  # pyright: ignore[reportCallIssue]
            (
                callback.funccalls,
                callback.best_evaluations,
            )
        ),
        delimiter=",",
        header="evals, best",
    )
    logger.info(f"{idx}: done with CMA-ES")


def run_bfgs(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(Elliptic.fun)
    callback = BFGSBestValueEvalCounterCallback(MAXEVALS)
    try:
        minimize(
            counter,
            x.copy(),
            method="BFGS",
            callback=lambda optres: callback(optres, counter),
        )

    finally:
        np.savetxt(
            BFGS_RESULT_DIR / f"{idx}.csv",
            np.column_stack((callback.funccalls, callback.best_evaluations)),
            delimiter=",",
            header="evals, best",
        )
    logger.info(f"{idx}: done with BFGS")


def run_lbfgs(x: np.ndarray, seed: int, idx: int):
    counter = EvalCounter(Elliptic.fun)
    callback = BFGSBestValueEvalCounterCallback(MAXEVALS)
    try:
        minimize(
            counter,
            x.copy(),
            method="L-BFGS-B",
            callback=lambda optres: callback(optres, counter),
        )

    finally:
        np.savetxt(
            RESULT_DIR / "bfgs" / f"{idx}.csv",
            np.column_stack((callback.funccalls, callback.best_evaluations)),
            delimiter=",",
            header="evals, best",
        )
    logger.info(f"{idx}: done with BFGS")


def run_n_cmaes_iterations(
    x: np.ndarray, seed: int, n: int, callback: CMAExperimentCallback
):
    inopts = {"popsize": 4 * len(x), "maxfevals": MAXEVALS, "seed": seed}
    es = CMAEvolutionStrategy(x, 1, inopts=inopts)
    while not es.stop():
        for _ in range(n):
            es.tell(*es.ask_and_eval(Elliptic))
            callback(es)
    return callback, es


def run_linesearch(x: np.ndarray, seed: int, idx: int, switch_after: int):
    callback = BestValueEvalCounterCallback()
    cma_results, es = run_n_cmaes_iterations(x, seed, switch_after, callback)

    fn = Elliptic.fun

    d = es.C @ gradient_central(fn, es.mean)
    xa, xb, xc, fa, fb, fc, funccalls = bracket(fn, maxiter=2000)
    es.countevals += funccalls

    solution, fval, funccalls = golden(fn, brack=(xa, xb, xc), full_output=True)
    es.countevals += funccalls

    callback.funccalls.append(es.countevals)
    callback.evaluations.append(fn(solution))

    assert callback.best is not None
    if (fval := fn(solution)) < callback.best[1]:
        callback.best = solution, fval
    callback.best_evaluations.append(callback.best[1])
    return callback


def run_cma_bfgs(x: np.ndarray, seed: int, idx: int, switch_after: int):
    callback = BestValueEvalCounterCallback()
    cma_results, es = run_n_cmaes_iterations(x, seed, switch_after, callback)
    cma_results = cast(BestValueEvalCounterCallback, cma_results)

    counter = EvalCounter(Elliptic.fun)
    callback = BFGSBestValueEvalCounterCallback(MAXEVALS - es.countevals)

    try:
        minimize(
            counter,
            es.mean,
            method="BFGS",
            callback=lambda optres: callback(optres, counter),
            options={
                "hess_inv0": es.C,
            },
        )

    finally:
        all_funccalls = np.concatenate([cma_results.funccalls, callback.funccalls])
        all_best_evaluations = np.concatenate(
            [cma_results.best_evaluations, callback.best_evaluations]
        )
        np.savetxt(
            CMABFGS_RESULT_DIR / f"{idx}.csv",
            np.column_stack((all_funccalls, all_best_evaluations)),
            delimiter=",",
            header="evals, best",
        )


def plot_results(result_path: str):

    fig = plt.figure()
    postscripts = ["vanilla", "bfgs", "lbfgs", "linesearch", "cmabfgs"]

    label_to_dirs = {
        "vanilla CMA-ES": VANILLA_RESULT_DIR,
        "vanilla BFGS": BFGS_RESULT_DIR,
        "vanilla L-BFGS": LBFGS_RESULT_DIR,
        "CMA-ES + linesearch": LINESEARCH_RESULT_DIR,
        "CMA-ES + BFGS": CMABFGS_RESULT_DIR,
    }

    for label, dir in label_to_dirs.items():
        x, y = load_and_interpolate_results(str(dir))
        sns.lineplot(
            x=x,
            y=y,
            label=label,
            ax=fig.gca(),
        )

    plt.title(f"Funkcja pokrzywiona w {DIM} wymiarach")
    plt.xlabel("Liczba ewaluacji f. celu")
    plt.ylabel("Najlepsze znalezione rozwiązanie")
    plt.yscale("log")
    plt.savefig(RESULT_DIR / f"plot.png")


def single_run(idx: int):
    seed: int = prime(idx)  # pyright: ignore[reportAssignmentType]
    rng = np.random.default_rng(seed)
    x = (rng.random(DIM) - 0.5) * 2 * BOUNDS

    run_vanilla(x, seed, idx)
    run_bfgs(x, seed, idx)
    run_lbfgs(x, seed, idx)
    run_linesearch(x, seed, idx, SWITCH_AFTER)
    run_cma_bfgs(x, seed, idx, SWITCH_AFTER)


def main():
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(single_run, range(1, NUM_RUNS + 1))


if __name__ == "__main__":
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR / "bfgs", exist_ok=True)
    os.makedirs(RESULT_DIR / "cma", exist_ok=True)

    main()
    plot_results(str(RESULT_DIR))
