from loguru import logger
import cma
from scipy.optimize import golden
from constants import PLOT_PATH
from funs import Elliptic, OptFun, ShiftedRastrigin, Rosen, Sphere
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util import distance_from_optimum


def one_dim(fun, x, d):
    """Gimmick to make a multdimensional function 1dim
    with a set direction d"""

    def wrapper(alpha):
        return fun.fun(x + alpha * d)

    return wrapper


DIMENSIONS = 100
BOUNDS = 3

rng = np.random.default_rng(0)


def single_run(fun: OptFun, popsize: int | None = None):
    x = (rng.random(DIMENSIONS) - 0.5) * 2 * BOUNDS
    es = cma.CMAEvolutionStrategy(x, 1, inopts={"popsize": popsize} if popsize else {})
    es.optimize(fun.fun)

    max_cutoff = int(es.countiter * 0.5)
    step = 50
    cutoffs = range(10, max_cutoff, 10)
    logger.info("Cutoffs: {}", cutoffs)

    # reinitialize
    x = (rng.random(DIMENSIONS) - 0.5) * 2 * BOUNDS
    es = cma.CMAEvolutionStrategy(x, 1, inopts={"popsize": popsize} if popsize else {})
    midpoint_distances, midpoint_values, golden_distances, golden_values = (
        [],
        [],
        [],
        [],
    )

    for i in range(cutoffs[-1] + 1):
        if es.stop():
            logger.info(f"Early stopping after {i} iterations")
            break
        es.tell(*es.ask_and_eval(fun.fun))

        if i in cutoffs:
            d = es.C @ fun.grad(es.mean)
            solution, fval, funcalls = golden(
                one_dim(fun, es.mean, d), full_output=True
            )
            logger.warning(f"Gradient value: {np.linalg.norm(fun.grad(es.mean))}")
            print(f"Golden solution: {solution} for cutoff {i}")
            # logger.info(f"Golden search took {funcalls} function calls at cutoff {i}")

            golden_solution = es.mean + solution * d
            actual_optimum = np.ones(DIMENSIONS) * fun.optimum

            golden_values.append(fun.fun(golden_solution))
            golden_distances.append(np.linalg.norm(golden_solution - actual_optimum))

            midpoint_values.append(fun.fun(es.mean))
            midpoint_distances.append(np.linalg.norm(es.mean - actual_optimum))

    save_dir = PLOT_PATH / "hybrid" / fun.name / str(DIMENSIONS)
    save_dir.mkdir(parents=True, exist_ok=True)

    x = np.array(cutoffs)

    fig, axes = plt.subplots(1, 2)
    plt.suptitle(f"{fun.name}: lambda={popsize}, dim={DIMENSIONS}")

    print(x.shape, len(golden_values), len(midpoint_values))
    sns.lineplot(
        x=x,
        y=golden_values,
        ax=axes[0],
        label="Golden search",
    )
    sns.lineplot(
        x=x,
        y=midpoint_values,
        ax=axes[0],
        label="CMA-ES midpoint",
    )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Cutoff/iterations")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Function value at the solution")

    sns.lineplot(
        x=x,
        y=golden_distances,
        ax=axes[1],
        label="Golden search",
    )
    sns.lineplot(
        x=x,
        y=midpoint_distances,
        ax=axes[1],
        label="CMA-ES midpoint",
    )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Cutoff/iterations")
    axes[1].set_ylabel("Distance")
    axes[1].set_yscale("log")
    axes[1].set_title("Distance from the optimum")
    plt.savefig(save_dir / f"popsize_{popsize}.png")
    # plt.show()


if __name__ == "__main__":
    for fun in (Sphere,):
        for size in (400,):
            single_run(fun, size)
