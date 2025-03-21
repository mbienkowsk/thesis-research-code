from dataclasses import dataclass, field
from typing import Callable
from cma import CMAEvolutionStrategy
import numpy as np
from scipy.optimize import golden
from enum import Enum

from funs import OptFun
from hybrid import one_dim

rng = np.random.default_rng(0)
DIMENSIONS = 100
INIT_BOUNDS = 3


class GradientType(Enum):
    PC = "pc"
    PC_C = "pc * C"
    ANALYTICAL_GRAD_C = "analytical gradient * C"
    DIVIDED_DIFFERENCE_C = "divided difference * C"


def lincmaes(
    x: np.ndarray,
    fun: OptFun,
    switch_interval: int,
    popsize: int | None = None,
    maxevals: int | None = None,
    gradient_type: GradientType = GradientType.ANALYTICAL_GRAD_C,
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
            es.tell(*es.ask_and_eval(fun.fun))
            evals_values.append(es.countevals)
            midpoint_values.append(fun.fun(es.mean))

        match gradient_type:
            case GradientType.PC:
                d = es.C @ es.pc
            case GradientType.PC_C:
                d = es.C @ es.pc * es.sigma
            case GradientType.ANALYTICAL_GRAD_C:
                d = es.C @ fun.grad(es.mean)
            case GradientType.DIVIDED_DIFFERENCE_C:
                raise NotImplementedError()

        # switch to linesearch
        d = es.C @ fun.grad(es.mean) if use_hessian else fun.grad(es.mean)

        solution, fval, funccalls = golden(one_dim(fun, es.mean, d), full_output=True)
        es.countevals += funccalls

        # Shift the mean
        solution = es.mean + solution * d
        es.mean = solution
        es.pc = 0

        # evals_values.append(es.countevals)
        # midpoint_values.append(fun.fun(es.mean))

        print(f"{es.countevals}: {np.linalg.norm(es.mean)}")

        print(f"{es.countevals} evaluations, looping over")

    print(f"{"Grad" if not use_hessian else "Hessian" } evals: {es.countevals}")
    return np.array(midpoint_values), np.array(evals_values)
