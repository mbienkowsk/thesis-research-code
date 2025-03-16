import cma
from loguru import logger
import numba
import numpy as np
from constants import PLOT_PATH
from funs import Elliptic, OptFun, ShiftedRastrigin, Rosen, Sphere
import matplotlib
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import multiprocessing
from pathlib import Path
from functools import partial

from util import distance_from_optimum, plot_angle

matplotlib.use("qtagg")

rng = np.random.default_rng(0)
DIMENSIONS = 100
START_SIGMA = 1
BOUNDS = 3


def plot_angles_and_distances(angles, distances, save_path: Path, save_plot=True):
    x = np.arange(1, len(angles) + 1)
    data = pd.DataFrame({"x": x, "cosine_similarity": angles, "distance": distances})
    fig, axes = plt.subplots(1, 2)
    sns.lineplot(data=data, x="x", y="cosine_similarity", ax=axes[0])
    sns.lineplot(data=data, x="x", y="distance", ax=axes[1])
    if save_plot:
        plt.savefig(save_path)
    else:
        plt.show()


def single_run(
    fun: OptFun,
    dim,
    sigma: float = START_SIGMA,
    bounds=BOUNDS,
    save_plot=True,
    plot_angles: bool = False,
):
    iter = 0

    x = (rng.random(dim) - 0.5) * 2 * bounds

    angles = []
    dists = []

    es = cma.CMAEvolutionStrategy(x, START_SIGMA)

    while not es.stop():

        angle = cosine_similarity(
            fun.grad(es.mean).reshape(1, -1), es.pc.reshape(1, -1)
        )
        if plot_angles:
            plot_angle(fun.grad(es.mean), es.pc, angle[0][0], es.mean)
        angles.append(angle[0][0])

        xi = es.C @ fun.grad(es.mean)  # pyright: ignore[reportOperatorIssue]
        dists.append(distance_from_optimum(xi, es.mean, np.zeros(dim)))

        es.tell(*es.ask_and_eval(fun.fun))

        iter += 1
        if iter % 1000 == 0:
            logger.info(f"Process {dim}: iter {iter}")

    save_path = PLOT_PATH / fun.name / f"cmaes_{dim}_{bounds}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_angles_and_distances(angles, dists, save_path, save_plot)

    logger.info(f"Process {dim}: finished")


def sphere_run(dim):
    single_run(Sphere, dim)


def eliptic_run(dim):
    single_run(Elliptic, dim)


def main():
    x = (rng.random(30) - 0.5) * 2 * BOUNDS
    es = cma.CMAEvolutionStrategy(x, START_SIGMA, inopts={"popsize": 120})
    # es.optimize(cma.ff.rosen)
    es.optimize(ShiftedRastrigin.fun)
    print(es.mean)


if __name__ == "__main__":
    main()
