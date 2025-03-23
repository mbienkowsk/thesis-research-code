from typing import Callable
import cma
import numba
import numpy as np
from dataclasses import dataclass


@dataclass
class OptFun:
    fun: Callable
    grad: Callable
    name: str
    optimum: int  # single input, multiplied by num of dims


@numba.njit
def elliptic(x):
    n = len(x)
    rv = 0
    for i in range(n):
        rv += 10 ** (6 * i / (n - 1)) * x[i] ** 2

    return rv


@numba.njit
def elliptic_grad(x):
    n = len(x)
    rv = np.zeros(n)
    for i in range(n):
        rv[i] = 2 * 10 ** (6 * i / (n - 1)) * x[i]
    return rv


Elliptic = OptFun(elliptic, elliptic_grad, "Elliptic", 0)


@numba.njit
def sphere(x):
    return np.sum(x**2)


@numba.njit
def sphere_grad(x):
    return 2 * x


Sphere = OptFun(sphere, sphere_grad, "Sphere", 0)

Rosen = OptFun(cma.ff.rosen, cma.ff.grad_rosen, "Rosenbrock", 1)


@numba.njit
def shifted_rastrigin_grad(x):
    return 2 * (x - 100) + 20 * np.sin(2 * np.pi * (x - 100))


def shifted_rastrigin(x):
    """Rastrigin test objective function"""
    if not np.isscalar(x[0]):
        N = len(x[0])
        return [
            10 * N + sum((xi - 100) ** 2 - 10 * np.cos(2 * np.pi * (xi - 100)))
            for xi in x
        ]
        # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
    N = len(x)
    return 10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x))


ShiftedRastrigin = OptFun(
    shifted_rastrigin, shifted_rastrigin_grad, "Shifted Rastrigin", 100
)


@numba.jit
def rastrigin_grad(x):
    return 2 * x + 20 * np.sin(2 * np.pi * x)


Rastrigin = OptFun(cma.ff.rastrigin, rastrigin_grad, "Rastrigin", 0)
