from loguru import logger
from opfunu.cec_based.cec import CecBenchmark
from cma import CMAEvolutionStrategy
import numpy as np
from scipy.optimize import bracket, golden
from enum import Enum
from scipy.differentiate import derivative

from funs import OptFun
from util import CMAResult, get_function, gradient_central

rng = np.random.default_rng(0)


def one_dim(fun: OptFun | CecBenchmark, x, d):
    """Gimmick to make a multdimensional function 1dim
    with a set direction d"""

    def wrapper(alpha):
        return get_function(fun)(x + alpha * d)

    return wrapper


class CMAVariation(Enum):
    VANILLA = "vanilla"
    PC = "pc"
    PC_C = "C * pc"
    ANALYTICAL_GRAD = "analytical gradient"
    ANALYTICAL_GRAD_C = "C * analytical gradient"
    CENTRAL_DIFFERENCE_C = "C * central difference"
    FORWARD_DIFFERENCE_C = "C * forward difference"


def lincmaes(
    x: np.ndarray,
    fun: OptFun | CecBenchmark,
    switch_interval: int,
    popsize: int,
    maxevals: int | None = None,
    gradient_type: CMAVariation = CMAVariation.ANALYTICAL_GRAD_C,
    gradient_cost: int = 0,
    seed: int = 0,
) -> CMAResult:
    midpoint_values = []
    evals_values = []
    best_values = []

    inopts = {}
    if popsize:
        inopts["popsize"] = popsize
    if maxevals:
        inopts["maxfevals"] = maxevals
    if seed:
        inopts["seed"] = seed

    es = CMAEvolutionStrategy(x, 1, inopts=inopts)

    while not es.stop():

        for i in range(switch_interval):
            f = get_function(fun)
            es.tell(*es.ask_and_eval(f))
            evals_values.append(es.countevals)
            midpoint_values.append(f(es.mean))
            best_values.append(f(es.best.x))

        match gradient_type:
            case CMAVariation.PC:
                d = es.pc

            case CMAVariation.PC_C:
                d = es.C @ es.pc  # pyright: ignore[reportOperatorIssue]

            case CMAVariation.ANALYTICAL_GRAD_C:
                if isinstance(fun, CecBenchmark):
                    raise ValueError(
                        "CecBenchmark does not support analytical gradient"
                    )
                es.countevals += gradient_cost
                d = es.C @ fun.grad(es.mean)

            case CMAVariation.ANALYTICAL_GRAD:
                if isinstance(fun, CecBenchmark):
                    raise ValueError(
                        "CecBenchmark does not support analytical gradient"
                    )
                es.countevals += gradient_cost
                d = fun.grad(es.mean)

            case CMAVariation.CENTRAL_DIFFERENCE_C:
                es.countevals += 2 * len(es.mean)
                d = es.C @ gradient_central(get_function(fun), es.mean)

            case CMAVariation.FORWARD_DIFFERENCE_C:
                es.countevals += len(es.mean)
                d = es.C @ derivative(
                    get_function(fun), es.mean
                )  # pyright: ignore[reportOperatorIssue]

            case _:
                raise ValueError("Vanilla should not be passed to lincmaes")

        try:
            fn = one_dim(fun, es.mean, d)
            xa, xb, xc, fa, fb, fc, funccalls = bracket(fn, maxiter=2000)
            es.countevals += funccalls
            solution, fval, funccalls = golden(fn, brack=(xa, xb, xc), full_output=True)
            es.countevals += funccalls

        except RuntimeError:
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
