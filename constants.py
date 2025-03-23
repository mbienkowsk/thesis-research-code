from pathlib import Path

from funs import Rastrigin, ShiftedRastrigin, Sphere, Rosen, Elliptic


PLOT_PATH = Path("plots/")
INIT_BOUNDS = 3
ALL_FUNS = (Rastrigin, ShiftedRastrigin, Sphere, Rosen, Elliptic)
