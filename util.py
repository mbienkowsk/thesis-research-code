from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import numba
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@numba.njit
def distance_from_optimum(xi, mean, optimum: np.ndarray):
    xi /= np.linalg.norm(xi)
    diff = optimum - mean
    projection_scalar = diff @ xi

    if projection_scalar < 0:
        # Optimum in the other direction
        return np.linalg.norm(diff)
    else:
        # Optimum in the same direction
        return np.linalg.norm(optimum - mean + projection_scalar * xi)


def plot_angle(grad, pc, angle, mean):
    def r(x):
        return np.round(x, 2)

    normalized_grad = grad / np.linalg.norm(grad)
    normalized_pc = pc / np.linalg.norm(pc)

    sns.lineplot(x=[0, normalized_grad[0]], y=[0, normalized_grad[1]], label="grad")
    sns.lineplot(x=[0, normalized_pc[0]], y=[0, normalized_pc[1]], label="pc")

    plt.title(f"grad: {r(grad)}\n pc: {r(pc)}\n angle: {r(angle)}\nat point {r(mean)}")
    plt.show()


@dataclass
class CMAResult:
    midpoint_values: np.ndarray
    best_values: np.ndarray
    nums_evals: np.ndarray

    @staticmethod
    def highest_eval_count(results: Iterable[list[CMAResult]]):
        return max(r.nums_evals[-1] for result in results for r in result)


@dataclass
class AggregatedCMAResult:
    midpoint_values: list[np.ndarray]
    best_values: list[np.ndarray]
    nums_evals: list[np.ndarray]

    @classmethod
    def from_results(cls, results: Iterable[CMAResult]):
        return cls(
            [r.midpoint_values for r in results],
            [r.best_values for r in results],
            [r.nums_evals for r in results],
        )


@dataclass
class InterpolatedCMAResult:
    x: np.ndarray
    midpoint_values: np.ndarray
    best_values: np.ndarray

    @classmethod
    def from_aggregated_results(cls, results: AggregatedCMAResult, maxevals: int):
        shortest_y = min(
            min(len(r) for r in results.midpoint_values),
            min(len(r) for r in results.best_values),
        )
        x = np.linspace(0, maxevals, shortest_y)

        return cls(
            x,
            np.mean(
                np.array(
                    [
                        np.interp(x, e, v)
                        for v, e in zip(results.midpoint_values, results.nums_evals)
                    ]
                ),
                axis=0,
            ),
            np.mean(
                np.array(
                    [
                        np.interp(x, e, v)
                        for v, e in zip(results.best_values, results.nums_evals)
                    ]
                ),
                axis=0,
            ),
        )

    @classmethod
    def from_results(cls, results: Iterable[CMAResult], maxevals: int):
        return cls.from_aggregated_results(
            AggregatedCMAResult.from_results(results), maxevals
        )

    def plot(self, axes: list[plt.Axes], label: str):
        axes[0].plot(self.x, self.midpoint_values, label=f"{label}")
        axes[1].plot(self.x, self.best_values, label=f"{label}")
