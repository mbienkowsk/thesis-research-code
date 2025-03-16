from dataclasses import dataclass, field
from typing import Callable
from cma import CMAEvolutionStrategy
import numpy as np
from scipy.optimize import golden

from funs import OptFun
from hybrid import one_dim

rng = np.random.default_rng(0)
DIMENSIONS = 100
INIT_BOUNDS = 3


def lincmaes(
    x: np.ndarray,
    fun: OptFun,
    switch_interval: int,
    popsize: int | None = None,
    maxevals: int | None = None,
    use_hessian: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint_values = []
    evals_values = []

    inopts = {}
    if popsize:
        inopts["popsize"] = popsize
    if maxevals:
        inopts["maxfevals"] = maxevals

    es = CMAEvolutionStrategy(x, 1, inopts=inopts)

    while not es.stop():

        for i in range(switch_interval):
            evals_values.append(es.countevals)
            midpoint_values.append(fun.fun(es.mean))
            es.tell(*es.ask_and_eval(fun.fun))

        # switch to linesearch
        d = es.C @ fun.grad(es.mean) if use_hessian else fun.grad(es.mean)

        solution, fval, funccalls = golden(one_dim(fun, es.mean, d), full_output=True)
        es.countevals += funccalls

        # Shift the mean
        solution = es.mean + solution * d
        print(
            f"Shifting solution {np.linalg.norm(es.mean)} -> {np.linalg.norm(solution)}"
        )
        es.mean = solution
        print("New mean", np.linalg.norm(es.mean))

        evals_values.append(es.countevals)
        midpoint_values.append(fun.fun(es.mean))

        es.tell(*es.ask_and_eval(fun.fun))
        print(f"{es.countevals}: {np.linalg.norm(es.mean)}")

        print(f"{es.countevals} evaluations, looping over")

    print(f"{"Grad" if not use_hessian else "Hessian" } evals: {es.countevals}")
    return np.array(midpoint_values), np.array(evals_values)
