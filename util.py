from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable
import numba
import numpy as np
from opfunu.cec_based.cec import CecBenchmark
import seaborn as sns
import matplotlib.pyplot as plt

from funs import OptFun


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


def gradient_forward(func: Callable, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """
    Calculate the gradient of a function at point x using forward difference.

    Parameters:
    -----------
    func : Callable
        Function to differentiate. Should take a numpy array and return a scalar.
    x : np.ndarray
        Point at which to calculate the gradient.
    h : float, optional
        Step size for finite difference, default 1e-6.

    Returns:
    --------
    np.ndarray
        Gradient vector of the function at point x.
    """
    n = len(x)
    grad = np.zeros_like(x, dtype=float)
    f0 = func(x)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        grad[i] = (func(x_plus) - f0) / h

    return grad


def gradient_central(func: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """
    Calculate the gradient of a function at point x using central difference.

    Parameters:
    -----------
    func : Callable
        Function to differentiate. Should take a numpy array and return a scalar.
    x : np.ndarray
        Point at which to calculate the gradient.
    h : float, optional
        Step size for finite difference, default 1e-6.

    Returns:
    --------
    np.ndarray
        Gradient vector of the function at point x.
    """
    n = len(x)
    grad = np.zeros_like(x, dtype=float)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)

    return grad


def get_function(f: CecBenchmark | OptFun):
    """Common interface for CEC and OptFun functions."""
    if isinstance(f, CecBenchmark):
        return f.evaluate
    else:
        return f.fun
