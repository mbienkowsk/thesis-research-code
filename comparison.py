from collections.abc import Iterable
import multiprocessing
from pathlib import Path
from loguru import logger
from opfunu.cec_based.cec import CecBenchmark
from constants import ALL_FUNS, INIT_BOUNDS, PLOT_PATH
from funs import Elliptic, OptFun, ShiftedRastrigin, Rosen, Sphere, Rastrigin
from lincmaes import CMAVariation
from util import CMAResult, InterpolatedCMAResult
from wrapper import eswrapper
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def average_interpolated_values(values, evals, maxevals):
    """Interpolates values to the same length and averages them.
    Returns both the x values and the y values to later plot."""

    shortest = min(len(v) for v in values)

    x = np.linspace(0, maxevals, shortest)

    return x, np.mean(
        np.array([np.interp(x, e, v) for v, e in zip(values, evals)]), axis=0
    )


def single_comparison(
    fun: OptFun | CecBenchmark,
    dims: int,
    popsize: int,
    maxevals: int,
    line_interval: int,
    average_from: int = 25,
    save_plot: bool = True,
    variations_to_test=(
        CMAVariation.VANILLA,
        CMAVariation.PC_C,
        CMAVariation.CENTRAL_DIFFERENCE_C,
        CMAVariation.FORWARD_DIFFERENCE_C,
    ),
):
    results: dict = {var: [] for var in variations_to_test}

    for _ in range(average_from):
        x = (rng.random(dims) - 0.5) * 2 * INIT_BOUNDS

        for variation in variations_to_test:

            results[variation].append(
                eswrapper(
                    x=x,
                    fun=fun,
                    popsize=popsize,
                    variation=variation,
                    line_search_interval=line_interval,
                    maxevals=maxevals,
                )
            )

    highest_eval_count = CMAResult.highest_eval_count(results.values())
    interpolated_results = {
        var: InterpolatedCMAResult.from_results(results[var], highest_eval_count)
        for var in results.keys()
    }
    save_dir = PLOT_PATH / f"quality_comparison_avg_{average_from}" / fun.name
    filename = f"dim_{dims}_k_{line_interval // dims}.png"
    plot_interpolated_results(
        ((v, k.value) for k, v in interpolated_results.items()),
        fun.name,
        dims,
        line_interval // dims,
        popsize,
        save_plot,
        save_dir / filename,
    )


def plot_interpolated_results(
    results: Iterable[tuple[InterpolatedCMAResult, str]],
    fun_name: str,
    dims: int,
    k: int,
    popsize: int,
    save_plot: bool,
    plot_path: Path | None,
):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.2)
    for result, label in results:
        result.plot(axes, label)
    axes[0].legend()
    axes[1].legend()

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("Midpoint values vs fun evaluations")
    axes[1].set_title("Best value vs fun evaluations")
    plt.suptitle(f"Function: {fun_name}, dimensions: {dims}, k: {k}, lambda: {popsize}")

    if save_plot:
        if plot_path is None:
            raise ValueError("No plot path provided")
        logger.warning("Saving plot")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
    else:
        plt.show()
    plt.clf()


def single_comparison_wrapper(fun: OptFun, dim: int, k: int, avg_from: int = 25):
    popsize = 4 * dim
    maxevals = 1000 * popsize
    line_cmaes_interval = k * dim
    single_comparison(fun, dim, popsize, maxevals, line_cmaes_interval, avg_from)


def run_all(
    dims: tuple[int] = (50,),
    funs: tuple = ALL_FUNS,
    ks: tuple = (1, 2, 3, 4),
    avg_from: int = 25,
):
    dims = (50,)
    funs = (Rastrigin, ShiftedRastrigin, Sphere, Rosen, Elliptic)
    ks = (1, 2, 3, 4)
    avg_from = 50

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(
            single_comparison_wrapper,
            [(fun, dim, k, avg_from) for fun in funs for dim in dims for k in ks],
        )


if __name__ == "__main__":
    run_all((30,), (Rastrigin,), (3, 4), avg_from=10)
