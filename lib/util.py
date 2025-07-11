from __future__ import annotations

import glob
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, override

import matplotlib.pyplot as plt
import numba
import numpy as np
import seaborn as sns
from cma.evolution_strategy import CMAEvolutionStrategy
from loguru import logger
from opfunu.cec_based.cec import CecBenchmark
from scipy.optimize import OptimizeResult

from .funs import OptFun

if TYPE_CHECKING:
    from lincmaes import CMAVariation


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
class StepSizeResult:
    x: np.ndarray
    golden_step_sizes: np.ndarray
    regular_step_sizes: np.ndarray


@dataclass
class BaseResult:
    fun: OptFun | CecBenchmark
    dim: int
    k: int | None
    grad_variation: "CMAVariation"


@dataclass
class CMAResult(BaseResult):
    midpoint_values: np.ndarray
    best_values: np.ndarray
    nums_evals: np.ndarray

    @staticmethod
    def highest_eval_count(results: Iterable[list[CMAResult]]):
        return max(r.nums_evals[-1] for result in results for r in result)


@dataclass
class AggregatedCMAResult(BaseResult):
    midpoint_values: list[np.ndarray]
    best_values: list[np.ndarray]
    nums_evals: list[np.ndarray]

    @classmethod
    def from_results(cls, results: list[CMAResult]):
        fun = results[0].fun
        dim = results[0].dim
        k = results[0].k
        grad_variation = results[0].grad_variation

        return cls(
            fun=fun,
            dim=dim,
            k=k,
            grad_variation=grad_variation,
            midpoint_values=[r.midpoint_values for r in results],
            best_values=[r.best_values for r in results],
            nums_evals=[r.nums_evals for r in results],
        )


@dataclass
class InterpolatedCMAResult(BaseResult):
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
            fun=results.fun,
            dim=results.dim,
            k=results.k,
            grad_variation=results.grad_variation,
            x=x,
            midpoint_values=np.mean(
                np.array(
                    [
                        np.interp(x, e, v)
                        for v, e in zip(results.midpoint_values, results.nums_evals)
                    ]
                ),
                axis=0,
            ),
            best_values=np.mean(
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
    def from_results(cls, results: list[CMAResult], maxevals: int):
        return cls.from_aggregated_results(
            AggregatedCMAResult.from_results(results), maxevals
        )

    def plot(self, axes: list[plt.Axes], label: str):
        axes[0].plot(self.x, self.midpoint_values, label=f"{label}")
        axes[1].plot(self.x, self.best_values, label=f"{label}")


def gradient_forward(func: Callable, x: np.ndarray, h: float = 1e-3) -> np.ndarray:
    n = len(x)
    grad = np.zeros_like(x, dtype=float)
    f0 = func(x)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        grad[i] = (func(x_plus) - f0) / h

    return grad


def gradient_central(func: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
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


class CMAExperimentCallback(ABC):
    @abstractmethod
    def __call__(self, es: CMAEvolutionStrategy):
        pass


def average_interpolated_values(values, evals, maxevals):
    """Interpolates values to the same length and averages them.
    Returns both the x values and the y values to later plot."""

    shortest = min(len(v) for v in values)

    x = np.linspace(0, maxevals, shortest)

    return x, np.mean(
        np.array([np.interp(x, e, v) for v, e in zip(values, evals)]), axis=0
    )


class EvalCounter(OptFun):
    """A wrapper to count the number of evaluations"""

    fun: Callable
    nfev: int
    best: tuple[np.ndarray, float] | None

    def __init__(self, fun: Callable):
        self.fun = fun
        self.nfev = 0
        self.best = None

    def __call__(self, x):
        self.nfev += 1
        y = self.fun(x)

        if self.best is None or y < self.best[1]:
            self.best = (x.copy(), y)

        return y


@dataclass
class BestValueEvalCounterCallback(CMAExperimentCallback):
    funccalls: list[int] = field(default_factory=list)
    evaluations: list[float] = field(default_factory=list)
    best_evaluations: list[float] = field(default_factory=list)
    best: tuple[np.ndarray, float] | None = field(default=None)

    @override
    def __call__(self, es):
        self.funccalls.append(es.countevals)
        self.evaluations.append(es.best.f)

        if self.best is None or es.best.f < self.best[1]:
            self.best = (es.mean.copy(), es.best.f)
            logger.info(f"New best value: {self.best[1]} at {self.best[0]}")

        self.best_evaluations.append(self.best[1])


@dataclass
class BFGSBestValueEvalCounterCallback(CMAExperimentCallback):
    maxevals: int
    counter: EvalCounter
    funccalls: list[int] = field(default_factory=list)
    evaluations: list[float] = field(default_factory=list)
    best_evaluations: list[float] = field(default_factory=list)
    best: tuple[np.ndarray, float] | None = field(default=None)

    @override
    def __call__(self, intermediate_result: OptimizeResult):
        if self.counter.nfev > self.maxevals:
            raise StopIteration()

        self.funccalls.append(self.counter.nfev)
        self.evaluations.append(intermediate_result.fun)

        if not self.best or intermediate_result.fun < self.best[1]:
            self.best = (intermediate_result.x, intermediate_result.fun)

        self.best_evaluations.append(self.best[1])
        logger.info(
            f"Current best value: {self.best[1]}"  # pyright: ignore[reportOptionalSubscript]
        )


def load_results_from_csv(path: str):
    """Given a csv of x,y pairs, load it and return two arrays"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def load_results_from_directory(dir_path: str):
    """Given a directory, load all CSVs and return lists of x and y arrays."""
    xx = []
    yy = []
    for csv_file in glob.glob(f"{dir_path}/*.csv"):
        x, y = load_results_from_csv(csv_file)
        xx.append(x)
        yy.append(y)
    return xx, yy


def load_and_interpolate_results(dir_path: str):
    """Given a directory with csv x/y pairs, load all of them and return the interpolated values"""
    xx, yy = load_results_from_directory(dir_path)
    xmax = max(x[-1] for x in xx)
    return average_interpolated_values(yy, xx, xmax)


def one_dim(fun: OptFun | CecBenchmark, x, d):
    """Gimmick to make a multdimensional function 1dim
    with a set direction d"""

    def wrapper(alpha):
        return get_function(fun)(x + alpha * d)

    return wrapper
