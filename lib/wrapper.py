import numpy as np
from cma import CMAEvolutionStrategy
from opfunu.cec_based.cec import CecBenchmark

from .constants import DEFAULT_CMA_OPTIONS
from .funs import OptFun
from .lincmaes import CMAVariation, lincmaes
from .util import CMAExperimentCallback, CMAResult, get_function


def eswrapper(
    x: np.ndarray,
    fun: OptFun | CecBenchmark,
    popsize: int,
    maxevals: int,
    variation: CMAVariation = CMAVariation.VANILLA,
    line_search_interval: int | None = None,
    gradient_cost: int = 0,
    seed: int = 0,
    callback: CMAExperimentCallback | None = None,
) -> CMAResult:
    """Wraps all variations of the CMA-ES into a single function with a common interface."""

    if variation != CMAVariation.VANILLA:
        assert line_search_interval is not None, "Line search interval must be set."
        return lincmaes(
            x,
            fun,
            line_search_interval,
            popsize,
            maxevals,
            gradient_type=variation,
            gradient_cost=gradient_cost,
            seed=seed,
        )[0]

    midpoint_values = []
    evals_values = []
    best_values = []

    inopts = DEFAULT_CMA_OPTIONS.copy()
    if popsize:
        inopts["popsize"] = popsize
    if maxevals:
        inopts["maxfevals"] = maxevals
    if seed:
        inopts["seed"] = seed

    es = CMAEvolutionStrategy(x, 1, inopts=inopts)

    while not es.stop():
        f = get_function(fun)
        try:
            es.tell(*es.ask_and_eval(f))
        except ValueError:
            with open("error.csv", "a") as file:
                file.write(f"{fun.name},{es.countevals},{es.mean},{variation}\n")

        if callback is not None:
            callback(es)

        evals_values.append(es.countevals)
        midpoint_values.append(f(es.mean))
        best_values.append(f(es.best.x))

    return CMAResult(
        fun=fun,
        dim=len(x),
        k=None,
        grad_variation=variation,
        midpoint_values=np.array(midpoint_values),
        best_values=np.array(best_values),
        nums_evals=np.array(evals_values),
    )
