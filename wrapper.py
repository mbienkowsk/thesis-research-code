from loguru import logger
from cma import CMAEvolutionStrategy
from funs import OptFun
import numpy as np

from lincmaes import CMAVariation, lincmaes
from util import CMAResult


def eswrapper(
    x: np.ndarray,
    fun: OptFun,
    popsize: int | None = None,
    maxevals: int | None = None,
) -> CMAResult:
    """Wraps all variations of the CMA-ES into a single function with a common interface."""
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
        es.tell(*es.ask_and_eval(fun.fun))
        evals_values.append(es.countevals)
        midpoint_values.append(fun.fun(es.mean))
        best_values.append(fun.fun(es.best.x))

    print(f"Vanilla total evals: {es.countevals}")
    return CMAResult(
        np.array(midpoint_values), np.array(best_values), np.array(evals_values)
    )
