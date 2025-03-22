from dataclasses import dataclass, field
from typing import Callable

from loguru import logger
from cma import CMAEvolutionStrategy
import numpy as np
from scipy.optimize import bracket, golden
from enum import Enum

from funs import OptFun
from hybrid import one_dim
from util import CMAResult

rng = np.random.default_rng(0)
DIMENSIONS = 100
INIT_BOUNDS = 3


class CMAVariation(Enum):
    VANILLA = "vanilla"
    PC = "pc"
    PC_C = "C * pc"
    ANALYTICAL_GRAD = "analytical gradient"
    ANALYTICAL_GRAD_C = "C * analytical gradient"
    DIVIDED_DIFFERENCE_C = "C * divided difference"


def lincmaes(
    x: np.ndarray,
    fun: OptFun,
    switch_interval: int,
    popsize: int,
    maxevals: int | None = None,
    gradient_type: CMAVariation = CMAVariation.ANALYTICAL_GRAD_C,
) -> CMAResult:
    midpoint_values = []
    evals_values = []
    best_values = []

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
            best_values.append(fun.fun(es.best.x))

        match gradient_type:
            case CMAVariation.PC:
                d = es.pc

            case CMAVariation.PC_C:
                d = es.C @ es.pc  # pyright: ignore[reportOperatorIssue]

            case CMAVariation.ANALYTICAL_GRAD_C:
                d = es.C @ fun.grad(es.mean)

            case CMAVariation.ANALYTICAL_GRAD:
                d = fun.grad(es.mean)

            case CMAVariation.DIVIDED_DIFFERENCE_C:
                raise NotImplementedError()

            case _:
                raise ValueError("Vanilla should not be passed to lincmaes")

        try:
            fn = one_dim(fun, es.mean, d)
            xa, xb, xc, fa, fb, fc, funccalls = bracket(fn, maxiter=2000)
            es.countevals += funccalls
            solution, fval, funccalls = golden(fn, brack=(xa, xb, xc), full_output=True)
            es.countevals += funccalls

        except Exception:
            msg = f"Golden failed at iteration {es.countevals} for fun {fun.name}, variation {gradient_type}, k = {switch_interval // len(x)}"
            with open("golden_failed.txt", "a") as f:
                f.write(msg + "\n")
            logger.error(msg)
            continue

        # Shift the mean
        solution = es.mean + solution * d
        es.mean = solution
        es.pc = 0

    logger.info(
        f"{str(gradient_type)} evals for for fun {fun.name}, variation {gradient_type}, k = {switch_interval // len(x)}: {es.countevals}"
    )
    return CMAResult(
        np.array(midpoint_values), np.array(best_values), np.array(evals_values)
    )
