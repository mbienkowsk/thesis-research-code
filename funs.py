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
def cylinder(x):
    n = len(x)
    rv = 0
    for i in range(n):
        rv += 10 ** (6 * i / (n - 1)) * x[i] ** 2

    return rv


@numba.njit
def cylinder_grad(x):
    n = len(x)
    rv = np.zeros(n)
    for i in range(n):
        rv[i] = 2 * 10 ** (6 * i / (n - 1)) * x[i]
    return rv


Cylinder = OptFun(cylinder, cylinder_grad, "cylinder", 0)


@numba.njit
def sphere(x):
    return np.sum(x**2)


@numba.njit
def sphere_grad(x):
    return 2 * x


Sphere = OptFun(sphere, sphere_grad, "sphere", 0)

Rosen = OptFun(cma.ff.rosen, cma.ff.grad_rosen, "Rosenbrock", 1)
