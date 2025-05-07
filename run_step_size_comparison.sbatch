#!/bin/bash -l

# Name of the job
#SBATCH -J step_size_comparison

# One node, one task, one task per node
#SBATCH --ntasks=1

# Memory limit
#SBATCH --mem=1GB

# num of cpus per task, time limit HH:MM:SS
#SBATCH --cpus-per-task=25
#SBATCH --time=72:00:00
# Grant name
#SBATCH -A plgcmaes

# Partition (plgrid for 72 hours)
#SBATCH -p plgrid

# Output files
#SBATCH --output="out.log"
#SBATCH --error="err.log"

cd $SLURM_SUBMIT_DIR

# Load Python and activate your environment
module load ~/modules/3.13.2

echo "Creating venv"
python3 -m venv .venv
source .venv/bin/activate  

# Install dependencies
echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# Run the Python script
echo "Running the script"
python3 -m scripts.step_size_comparison
