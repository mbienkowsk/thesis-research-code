import multiprocessing

import numpy as np
from comparison import plot_interpolated_results
from constants import ALL_FUNS, INIT_BOUNDS, PLOT_PATH
from funs import OptFun
from lincmaes import CMAVariation
from util import CMAResult, InterpolatedCMAResult
from wrapper import eswrapper

rng = np.random.default_rng(0)


def single_run(fun: OptFun, dim: int, k: int, avg_from=25):
    popsize = 4 * dim
    maxevals = 1000 * popsize
    grad_cost_combs = (
        (CMAVariation.VANILLA, 0),
        (CMAVariation.ANALYTICAL_GRAD_C, 0),
        (CMAVariation.ANALYTICAL_GRAD_C, 1),
        (CMAVariation.ANALYTICAL_GRAD_C, dim),
    )
    results = {combo: [] for combo in grad_cost_combs}

    for _ in range(avg_from):
        x = (rng.random(dim) - 0.5) * 2 * INIT_BOUNDS
        for combo in grad_cost_combs:
            results[combo].append(
                eswrapper(
                    x=x,
                    fun=fun,
                    popsize=popsize,
                    variation=combo[0],
                    line_search_interval=k * dim,
                    maxevals=maxevals,
                    gradient_cost=combo[1],
                    seed=1,
                )
            )

        highest_eval_count = CMAResult.highest_eval_count(results.values())
        interpolated_results = {
            combo: InterpolatedCMAResult.from_results(
                results[combo], highest_eval_count
            )
            for combo in results.keys()
        }
        save_dir = PLOT_PATH / f"gradient_cost_comparison_avg_{avg_from}" / fun.name
        filename = f"dim_{dim}_k_{k}.png"
        plot_interpolated_results(
            (
                (val, f"{key[0].value} grad cost: {key[1]}")
                for (key, val) in interpolated_results.items()
            ),
            fun.name,
            dim,
            k,
            popsize,
            True,
            save_dir / filename,
        )


if __name__ == "__main__":

    dims = (30, 50)
    funs = ALL_FUNS
    ks = (1, 2, 3, 4)

    with multiprocessing.Pool(5) as pool:
        pool.starmap(
            single_run,
            [(fun, dim, k, 50) for fun in funs for dim in dims for k in ks],
        )
