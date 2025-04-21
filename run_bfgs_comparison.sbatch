#!/bin/bash -l
#SBATCH -J bfgs_comparison
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH -A plgcmaes
#SBATCH -p plgrid
#SBATCH --output="out.log"
#SBATCH --error="err.log"

cd $SLURM_SUBMIT_DIR

# Load Python and activate your environment
module load trytonp/python/3.12.0

echo "Creating venv"
python3 -m venv .venv
source .venv/bin/activate  # Or use conda if you prefer

# Install dependencies
echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# Run the Python script
echo "Running the script"
python3 -m scripts.bfgs_comparison
