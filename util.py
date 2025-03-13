import numba
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@numba.njit
def distance_from_optimum(xi, mean, optimum: np.ndarray):
    xi /= np.linalg.norm(xi)
    diff = optimum - mean
    projection_scalar = diff @ xi

    if projection_scalar < 0:
        # Optimum in the other direction
        return np.linalg.norm(diff)
    else:
        # Optimum in the same direction
        return np.linalg.norm(optimum - mean + projection_scalar * xi)


def plot_angle(grad, pc, angle, mean):
    def r(x):
        return np.round(x, 2)

    normalized_grad = grad / np.linalg.norm(grad)
    normalized_pc = pc / np.linalg.norm(pc)

    sns.lineplot(x=[0, normalized_grad[0]], y=[0, normalized_grad[1]], label="grad")
    sns.lineplot(x=[0, normalized_pc[0]], y=[0, normalized_pc[1]], label="pc")

    plt.title(f"grad: {r(grad)}\n pc: {r(pc)}\n angle: {r(angle)}\nat point {r(mean)}")
    plt.show()
