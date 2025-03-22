import multiprocessing
from loguru import logger
from constants import PLOT_PATH
from funs import Elliptic, OptFun, ShiftedRastrigin, Rosen, Sphere, Rastrigin
from lincmaes import CMAVariation, lincmaes
from util import AggregatedCMAResult, CMAResult, InterpolatedCMAResult
from wrapper import eswrapper
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

rng = np.random.default_rng(0)
INIT_BOUNDS = 3


def average_interpolated_values(values, evals, maxevals):
    """Interpolates values to the same length and averages them.
    Returns both the x values and the y values to later plot."""

    shortest = min(len(v) for v in values)

    x = np.linspace(0, maxevals, shortest)

    return x, np.mean(
        np.array([np.interp(x, e, v) for v, e in zip(values, evals)]), axis=0
    )


def single_comparison(
    fun: OptFun,
    dims: int,
    popsize: int,
    maxevals: int,
    line_interval: int,
    average_from: int = 25,
    save_plot: bool = True,
    variations_to_test=(
        CMAVariation.VANILLA,
        CMAVariation.PC,
        CMAVariation.ANALYTICAL_GRAD_C,
        CMAVariation.PC_C,
    ),
):
    results: dict = {var: [] for var in variations_to_test}

    for _ in range(average_from):
        x = (rng.random(dims) - 0.5) * 2 * INIT_BOUNDS

        if CMAVariation.VANILLA in variations_to_test:
            results[CMAVariation.VANILLA].append(eswrapper(x, fun, popsize, maxevals))

        for variation in variations_to_test:
            if variation == CMAVariation.VANILLA:
                continue  # TODO: unify the interface!

            results[variation].append(
                lincmaes(
                    x,
                    fun,
                    line_interval,
                    popsize,
                    maxevals,
                    gradient_type=variation,
                )
            )

    highest_eval_count = CMAResult.highest_eval_count(results.values())
    interpolated_results = {
        var: InterpolatedCMAResult.from_results(results[var], highest_eval_count)
        for var in results.keys()
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.2)
    for key, result in interpolated_results.items():
        result.plot(axes, key.value)
    axes[0].legend()
    axes[1].legend()

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("Midpoint values vs fun evaluations")
    axes[1].set_title("Best value vs fun evaluations")
    plt.suptitle(
        f"Function: {fun.name}, dimensions: {dims}, k: {line_interval // dims}, lambda: {popsize}"
    )

    save_dir = PLOT_PATH / "hybrid" / "comparison_new" / fun.name
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_plot:
        logger.warning("Saving plot")
        plt.savefig(save_dir / f"dim_{dims}_k_{line_interval // dims}.png")
    else:
        plt.show()
    plt.clf()


def single_comparison_wrapper(fun: OptFun, dim: int, k: int, avg_from: int = 25):
    popsize = 4 * dim
    maxevals = 1000 * popsize
    line_cmaes_interval = k * dim
    single_comparison(fun, dim, popsize, maxevals, line_cmaes_interval, avg_from)


def run_all():
    dims = (30, 50)
    # funs = (Rastrigin, ShiftedRastrigin, Sphere, Rosen, Elliptic)
    funs = (Rosen,)
    ks = (1, 2, 3, 4)

    with multiprocessing.Pool(6) as pool:
        pool.starmap(
            single_comparison_wrapper,
            [(fun, dim, k, 5) for fun in funs for dim in dims for k in ks],
        )


def main():
    dims = 30
    popsize = 4 * dims
    maxevals = 1000 * popsize
    line_cmaes_interval = 3 * dims

    # single_comparison(
    #     Sphere,
    #     dims,
    #     popsize,
    #     maxevals,
    #     line_cmaes_interval,
    #     25,
    #     False,
    #     variations_to_test=(
    #         CMAVariation.VANILLA,
    #         CMAVariation.ANALYTICAL_GRAD_C,
    #         CMAVariation.ANALYTICAL_GRAD,
    #     ),
    # )
    run_all()


if __name__ == "__main__":
    main()
