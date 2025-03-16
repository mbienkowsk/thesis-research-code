from cma import CMAEvolutionStrategy
from funs import OptFun
import numpy as np


def eswrapper(
    x: np.ndarray, fun: OptFun, popsize: int | None = None, maxevals: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Wraps the CMA-ES optimizer to have the same interface as lincmaes"""
    midpoint_values = []
    evals_values = []

    inopts = {}
    if popsize:
        inopts["popsize"] = popsize
    if maxevals:
        inopts["maxfevals"] = maxevals

    es = CMAEvolutionStrategy(x, 1, inopts=inopts)

    while not es.stop():
        evals_values.append(es.countevals)
        midpoint_values.append(fun.fun(es.mean))
        es.tell(*es.ask_and_eval(fun.fun))

    print(f"Vanilla total evals: {es.countevals}")
    return np.array(midpoint_values), np.array(evals_values)
