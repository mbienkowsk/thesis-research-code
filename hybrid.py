from loguru import logger
import seaborn as sns
import cma
import numpy as np
from constants import PLOT_PATH
from funs import OptFun, Rosen
from scipy.optimize import bracket, golden
from codetiming import Timer
from dataclasses import dataclass
from matplotlib import pyplot as plt

from util import distance_from_optimum


rng = np.random.default_rng(0)

DIMENSIONS = 30
START_SIGMA = 1
BOUNDS = 3


@dataclass
class SingleResult:
    fun_val: np.float64
    dist_from_opt: np.float64


def one_dim(fun, x, d):
    """Gimmick to make a multdimensional function 1dim
    with a set direction d"""

    def wrapper(alpha):
        return fun.fun(x + alpha * d)

    return wrapper


def single_run(fun: OptFun, stop_after: int, popsize: int | None = None):
    x = (rng.random(DIMENSIONS) - 0.5) * 2 * BOUNDS

    inopts = {"popsize": popsize} if popsize is not None else {}

    es = cma.CMAEvolutionStrategy(x, START_SIGMA, inopts=inopts)
    print(es.popsize)

    logger.info("Starting optimization using CMA-ES")

    for iter in range(stop_after):
        es.tell(*es.ask_and_eval(fun.fun))

    logger.info("Finished optimization using CMA-ES")

    d = es.C @ fun.grad(es.mean)

    one_dim_fun = one_dim(fun, es.mean, d)
    solution, fval, funcalls = golden(one_dim_fun, full_output=True)

    print(f"Value in optimum: {fval}")
    print(f"Function calls in optimum: {funcalls}")

    multidim_solution = es.mean + solution * d

    fun_val = fun.fun(multidim_solution)
    actual_optimum = np.ones(DIMENSIONS) * fun.optimum
    dist_from_opt = np.linalg.norm(multidim_solution - actual_optimum)

    print(f"Function value at the optimum from golden: { fun_val }")
    print("Distance of the solution from optimum: ", dist_from_opt)

    return SingleResult(fun_val, dist_from_opt)


def main():
    popsizes = (None, DIMENSIONS, 2 * DIMENSIONS, 4 * DIMENSIONS)
    stop_values = range(100, 2001, 200)
    FUN = Rosen

    results = {
        popsize: {
            stop_value: single_run(Rosen, stop_value, popsize)
            for stop_value in stop_values
        }
        for popsize in popsizes
    }

    save_dir = PLOT_PATH / "hybrid" / FUN.name / str(DIMENSIONS)
    save_dir.mkdir(parents=True, exist_ok=True)

    for popsize in popsizes:
        losses = np.array([results[popsize][stop].fun_val for stop in stop_values])
        dists = np.array([results[popsize][stop].dist_from_opt for stop in stop_values])
        x = np.array(stop_values)

        fig, axes = plt.subplots(1, 2)

        popsize = popsize if popsize is not None else 14  # TODO: take from cma
        plt.suptitle(f"Population size: {popsize} in {DIMENSIONS} dimensions")

        sns.lineplot(
            x=x,
            y=losses,
            ax=axes[0],
        )
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Loss")

        sns.lineplot(
            x=x,
            y=dists,
            ax=axes[1],
        )
        axes[1].set_yscale("log")
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Distance from optimum")

        plt.savefig(save_dir / f"popsize_{popsize}.png")
        plt.show()


if __name__ == "__main__":
    main()
