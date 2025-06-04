"""For every iteration, compares the distance of the mean to the optimum, the distance of the
line achieved by the C * gradient product, and the distance of the line achieved by the C * pc product
"""

import multiprocessing
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import cast, override

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sympy import prime

from lib import funs
from lib.lincmaes import CMAVariation
from lib.util import CMAExperimentCallback
from lib.wrapper import eswrapper

DIMS = 10
FUN = funs.Elliptic
RESULT_DIR = Path(f"results/direction_distance_comparison/dim_{DIMS}")
NUM_RUNS = 10
MAXEVALS = 4000 * DIMS


def calculate_distance(m: np.ndarray, d: np.ndarray, optimum: np.ndarray):
    v = optimum - m

    alpha_star = np.dot(v, d) / np.dot(d, d)

    closest_point = m + alpha_star * d

    distance = np.linalg.norm(closest_point - optimum)

    return distance


@dataclass
class DirectionDistanceCallback(CMAExperimentCallback):
    funccalls: list[int] = field(default_factory=list)
    mean_distances: list[float] = field(default_factory=list)
    pc_c_distances: list[float] = field(default_factory=list)
    grad_c_distances: list[float] = field(default_factory=list)
    evaluations: list[float] = field(default_factory=list)

    @override
    def __call__(self, es):
        d1 = es.C @ FUN.grad(es.mean)
        d2 = es.C @ es.pc  # pyright: ignore[reportOperatorIssue]
        optimum = FUN.optimum_for_dim(DIMS)

        distance_1 = cast(float, calculate_distance(es.mean, d1, optimum))
        distance_2 = cast(float, calculate_distance(es.mean, d2, optimum))
        mean_distance = cast(float, np.linalg.norm(es.mean - optimum))

        self.funccalls.append(es.countevals)
        self.grad_c_distances.append(distance_1)
        self.pc_c_distances.append(distance_2)
        self.mean_distances.append(mean_distance)
        self.evaluations.append(es.best.f)


def single_direction_dist_comp(idx: int):
    seed: int = cast(int, prime(idx))

    rng = np.random.default_rng(seed)

    x = (rng.random(DIMS) - 0.5) * 2 * 50
    popsize = 4 * DIMS
    callback = DirectionDistanceCallback()

    _ = eswrapper(
        x=x,
        fun=FUN,
        popsize=popsize,
        maxevals=MAXEVALS,
        variation=CMAVariation.VANILLA,
        seed=seed,
        callback=callback,
    )

    np.savetxt(
        RESULT_DIR / f"{idx}.csv",
        np.column_stack(
            (
                np.array(callback.funccalls),
                np.array(callback.mean_distances),
                np.array(callback.grad_c_distances),
                np.array(callback.pc_c_distances),
                np.array(callback.evaluations),
            )
        ),
        delimiter=",",
        header="func_calls,mean_distances,grad_c_distances,pc_c_distances,evaluations",
        comments="",
    )

    logger.info(f"Run {idx}: completed direction/distance comparison")


def aggregate_direction_distance_results(output_fname: str | None) -> pd.DataFrame:
    accumulator: dict[int, list[list[float]]] = defaultdict(list)

    for file in sorted(RESULT_DIR.glob("*.csv")):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            key = int(row["func_calls"])
            values = [
                row["mean_distances"],
                row["grad_c_distances"],
                row["pc_c_distances"],
                row["evaluations"],
            ]
            accumulator[key].append(values)  # pyright: ignore[reportArgumentType]

    aggregated = []
    for func_calls, values_list in sorted(accumulator.items()):
        arr = np.array(values_list)
        avg_values = np.mean(arr, axis=0)
        aggregated.append([func_calls] + avg_values.tolist())

    result_df = pd.DataFrame(
        aggregated,
        columns=[  # pyright: ignore[reportArgumentType]
            "func_calls",
            "mean_distances",
            "grad_c_distances",
            "pc_c_distances",
            "evaluations",
        ],
    )

    if output_fname:
        result_df.to_csv(RESULT_DIR / output_fname, index=False)
        logger.info(f"Aggregated results saved to {output_fname}")

    return result_df


def compute_best_so_far(df: pd.DataFrame) -> pd.DataFrame:
    best_df = df.copy()
    best_df["mean_distances"] = best_df["mean_distances"].cummin()
    best_df["grad_c_distances"] = best_df["grad_c_distances"].cummin()
    best_df["pc_c_distances"] = best_df["pc_c_distances"].cummin()
    return best_df


def plot_direction_distance_results(
    df: pd.DataFrame,
    title: str = "Distance to Optimum Over Time",
    output_fname: str | None = None,
):
    sns.set_theme(style="whitegrid", context="talk")

    _, axes = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    top = axes[0]

    top.plot(
        df["func_calls"], df["mean_distances"], label="‖mean − optimum‖", linewidth=2
    )
    top.plot(
        df["func_calls"], df["grad_c_distances"], label="‖proj(C * grad)‖", linewidth=2
    )
    top.plot(
        df["func_calls"], df["pc_c_distances"], label="‖proj(C * pc)‖", linewidth=2
    )
    top.set_yscale("log")
    top.set_ylabel("Distance to optimum (log scale)")
    top.set_title(title)
    top.legend()
    top.grid(True)

    bottom = axes[1]
    bottom.plot(
        df["func_calls"],
        df["evaluations"],
        label="Best-so-far f(x)",
        color="tab:orange",
        linewidth=2,
    )
    bottom.set_yscale("log")
    bottom.set_ylabel("Value of the best point")
    bottom.set_xlabel("Function evaluations")
    bottom.set_title("Convergence curve")
    bottom.legend()
    bottom.grid(True)

    plt.tight_layout()

    if output_fname:
        plt.savefig(RESULT_DIR / output_fname)
        logger.info(f"Plot saved to {output_fname}")
    else:
        plt.show()


def main():
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)

    with Pool(multiprocessing.cpu_count()) as pool:
        pool.map(single_direction_dist_comp, range(1, NUM_RUNS + 1))

    aggregated = aggregate_direction_distance_results("aggregated_results.csv")
    plot_direction_distance_results(
        aggregated,
        title=f"Distance to Optimum for {FUN.name} (Dim={DIMS})",
        output_fname="direction_distance_plot.png",
    )
    plot_direction_distance_results(
        compute_best_so_far(aggregated),
        title=f"Best distance to Optimum for {FUN.name} (Dim={DIMS})",
        output_fname="best_direction_distance_plot.png",
    )

    logger.info("All comparisons completed")


if __name__ == "__main__":
    main()
    logger.info(f"Step size comparison data saved to {RESULT_DIR}")
