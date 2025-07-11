from enum import Enum

import numpy as np
from cma import CMAEvolutionStrategy
from opfunu.cec_based.cec import CecBenchmark
from scipy.optimize import bracket, golden

from .funs import OptFun
from .util import (CMAExperimentCallback, CMAResult, StepSizeResult,
                   get_function, gradient_central, gradient_forward, one_dim)

rng = np.random.default_rng(0)


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
    get_step_information: bool = False,
    callback: CMAExperimentCallback | None = None,
) -> tuple[CMAResult, StepSizeResult | None]:
    midpoint_values = []
    evals_values = []
    best_values = []

    if get_step_information:
        golden_step_x, golden_step_sizes, regular_step_sizes = [], [], []

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
            # TODO: figure out why this even happens

            try:
                es.tell(*es.ask_and_eval(f))
                if callback is not None:
                    callback(es)

            except ValueError:
                with open("lincmaes_failed.csv", "a") as file:
                    file.write(
                        f"{fun.name},{es.countevals},{gradient_type},{switch_interval // len(x)}, {f(es.mean)}\n"
                    )
                    continue

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
                d = es.C @ gradient_forward(
                    get_function(fun), es.mean, 1e-3
                )  # pyright: ignore[reportOperatorIssue]

            case _:
                raise ValueError("Vanilla should not be passed to lincmaes")

        try:
            fn = one_dim(fun, es.mean, d)
            xa, xb, xc, fa, fb, fc, funccalls = bracket(fn, maxiter=2000)
            es.countevals += funccalls
            solution, fval, funccalls = golden(fn, brack=(xa, xb, xc), full_output=True)
            es.countevals += funccalls

            if get_step_information:
                golden_step_x.append(funccalls)
                golden_step_sizes.append(np.linalg.norm(solution - es.mean))
                regular_step_sizes.append(np.linalg.norm(es.sigma * es.delta))

            # Shift the mean
            solution = es.mean + solution * d
            es.mean = solution
            es.pc = np.zeros_like(solution)

        except RuntimeError:
            with open("golden_failed.csv", "a") as f:
                f.write(
                    f"bracket,{fun.name},{es.countevals},{gradient_type},{switch_interval // len(x)}\n"
                )
            continue

        except ValueError:
            with open("golden_failed.csv", "a") as f:
                f.write(
                    f"golden,{fun.name},{es.countevals},{gradient_type},{switch_interval // len(x)}\n"
                )
            continue

    result = CMAResult(
        fun=fun,
        dim=len(x),
        k=int(switch_interval / len(x)),
        grad_variation=gradient_type,
        midpoint_values=np.array(midpoint_values),
        best_values=np.array(best_values),
        nums_evals=np.array(evals_values),
    )

    ss_result = None
    if get_step_information:
        ss_result = StepSizeResult(
            x=np.array(golden_step_x),
            golden_step_sizes=np.array(golden_step_sizes),
            regular_step_sizes=np.array(regular_step_sizes),
        )

    return result, ss_result
