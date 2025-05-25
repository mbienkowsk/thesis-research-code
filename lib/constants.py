from pathlib import Path
from typing import Any

from .funs import Rastrigin, ShiftedRastrigin, Sphere, Rosen, Elliptic


PLOT_PATH = Path("plots/")
INIT_BOUNDS = 100
ALL_FUNS = (Rastrigin, ShiftedRastrigin, Sphere, Rosen, Elliptic)

DEFAULT_CMA_OPTIONS: dict[str, Any] = {
    "tolfun": 1e-9,
    "tolfunhist": 1e-9,
    "tolflatfitness": 3,
}
