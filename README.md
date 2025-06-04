# Research code for my thesis

## Topic: Improving the CMA-ES algorithm as a global optimization method

Heavy emphasis on research code - at this stage, this isn't really meant to be read after being written and forgotten.

## Repository structure

* every experiment lives in its own directory in `experiments/` directory except the ones in `archive/` that aren't relevant anymore and shouldn't clutter the main workspace.
* `lib/` contains everything that I try to make reusable across experiments
* `results/` contains plots and csvs from most of the performed experiments
* `pycma/` is a fork with reduced cyclometric complexity


