from funs import Elliptic, OptFun, ShiftedRastrigin, Rosen, Sphere, Rastrigin
from lincmaes import lincmaes
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
):
    all_vanilla_values, all_vanilla_evals = [], []
    all_hess_values, all_hess_evals = [], []
    all_grad_values, all_grad_evals = [], []

    for _ in range(average_from):
        x = (rng.random(dims) - 0.5) * 2 * INIT_BOUNDS
        v_values, v_evals = eswrapper(x, fun, popsize, maxevals)
        all_vanilla_values.append(v_values)
        all_vanilla_evals.append(v_evals)

        h_values, h_evals = lincmaes(
            x, fun, line_interval, popsize, maxevals, use_hessian=True
        )
        all_hess_values.append(h_values)
        all_hess_evals.append(h_evals)

        g_values, g_evals = lincmaes(
            x, fun, line_interval, popsize, maxevals, use_hessian=False
        )
        all_grad_values.append(g_values)
        all_grad_evals.append(g_evals)

    highest_eval_count = max(
        max(v[-1], h[-1], g[-1])
        for v, h, g in zip(all_vanilla_evals, all_hess_evals, all_grad_evals)
    )
    vanilla_x, vanilla_y = average_interpolated_values(
        all_vanilla_values, all_vanilla_evals, highest_eval_count
    )
    hess_x, hess_y = average_interpolated_values(
        all_hess_values, all_hess_evals, highest_eval_count
    )
    grad_x, grad_y = average_interpolated_values(
        all_grad_values, all_grad_evals, highest_eval_count
    )

    sns.lineplot(x=vanilla_x, y=vanilla_y, label="Vanilla CMA-ES")
    sns.lineplot(x=hess_x, y=hess_y, label="Line CMA-ES w/ Hessian approximation")
    sns.lineplot(
        x=grad_x, y=grad_y, label="Grad CMA-ES w/o Hessian approximation", ls="--"
    )
    plt.suptitle(
        f"Average value of {fun.name} at the midpoint after x evaluations of it."
        + f"\nlambda={popsize}, {dims} dimensions"
    )
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    dims = 30
    popsize = 4 * dims
    maxevals = 1000 * popsize
    line_cmaes_interval = 2 * dims

    single_comparison(Sphere, dims, popsize, maxevals, line_cmaes_interval, 10)
