"""Compares the step sizes performed by the regular CMA-ES and the golden search version."""

import multiprocessing
import os
from multiprocessing import Pool
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sympy import prime

from lib import funs
from lib.lincmaes import CMAVariation, lincmaes
from lib.util import average_interpolated_values

DIMS = 10
FUN = funs.Elliptic
RESULT_DIR = Path("results/step_size_comparison")
NUM_RUNS = 50
MAXEVALS = 4000 * DIMS

rng = np.random.default_rng(0)


def single_step_size_comp(idx: int):
    seed: int = cast(int, prime(idx))

    x = (rng.random(DIMS) - 0.5) * 2 * 5

    popsize = 4 * DIMS
    switch_interval = 1 * DIMS

    _, step_size_result = lincmaes(
        x=x,
        fun=FUN,
        switch_interval=switch_interval,
        popsize=popsize,
        maxevals=MAXEVALS,
        gradient_type=CMAVariation.ANALYTICAL_GRAD_C,
        seed=seed,
        get_step_information=True,
    )

    if step_size_result is not None:
        np.savetxt(
            RESULT_DIR / f"{idx}.csv",
            np.column_stack(
                (
                    step_size_result.x,
                    step_size_result.golden_step_sizes,
                    step_size_result.regular_step_sizes,
                )
            ),
            delimiter=",",
            header="func_calls,golden_step_size,regular_step_size",
            comments="",
        )

    logger.info(f"Run {idx}: completed step size comparison")


def interpolate_and_average_step_sizes():
    all_results = []
    for i in range(1, NUM_RUNS + 1):
        try:
            data = np.loadtxt(RESULT_DIR / f"{i}.csv", delimiter=",", skiprows=1)
            all_results.append(data)
        except Exception as e:
            logger.warning(f"Could not load file {i}.csv: {e}")

    if not all_results:
        logger.error("No results found to analyze")
        return None, None, None

    func_calls = [result[:, 0] for result in all_results if result.shape[0] > 0]
    golden_steps = [result[:, 1] for result in all_results if result.shape[0] > 0]
    regular_steps = [result[:, 2] for result in all_results if result.shape[0] > 0]

    max_calls = max(calls[-1] for calls in func_calls if len(calls) > 0)

    x, avg_golden = average_interpolated_values(golden_steps, func_calls, max_calls)

    _, avg_regular = average_interpolated_values(regular_steps, func_calls, max_calls)

    return x, avg_golden, avg_regular


def plot_step_sizes(x, golden_steps, regular_steps):

    plt.yscale("log")
    plt.plot(x, golden_steps, label="Golden Search", color="gold", linewidth=2)
    plt.plot(x, regular_steps, label="Regular CMA-ES", color="blue", linewidth=2)

    plt.xlabel("Function Evaluations")
    plt.ylabel("Step Size")
    plt.title(f"{FUN.name} Step Size Comparison (Dim={DIMS})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_dir = Path("plots/step_size_comparison")
    os.makedirs(plot_dir, exist_ok=True)

    plt.savefig(plot_dir / f"{FUN.name}_dim{DIMS}.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Step size plot saved to {plot_dir}/{FUN.name}_dim{DIMS}.png")


def main():
    """Run the step size comparison across multiple threads"""
    os.makedirs(RESULT_DIR, exist_ok=True)

    with Pool(multiprocessing.cpu_count()) as pool:
        pool.map(single_step_size_comp, range(1, NUM_RUNS + 1))

    logger.info("All step size comparisons completed")

    x, golden_steps, regular_steps = interpolate_and_average_step_sizes()

    if x is not None and golden_steps is not None and regular_steps is not None:
        plot_step_sizes(x, golden_steps, regular_steps)


if __name__ == "__main__":
    main()

    logger.info(f"Step size comparison data saved to {RESULT_DIR}")
