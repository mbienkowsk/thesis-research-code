#!/bin/bash -l
#SBATCH -J comparison_run
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH -A your-grant-id
#SBATCH -p plgrid
#SBATCH --output="out.log"
#SBATCH --error="err.log"

cd $SLURM_SUBMIT_DIR

# Load Python and activate your environment
module load python/3.12
source .venv/bin/activate  # Or use conda if you prefer

# Run the Python script
python bfgs_comparison.py
