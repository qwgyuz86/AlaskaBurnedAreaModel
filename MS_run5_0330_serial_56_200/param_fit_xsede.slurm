#!/bin/bash
#SBATCH --partition=shas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24	# number of tasks/threads for multiprocessing module
#SBATCH --account=rmacc-general
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siuyin.lee@ucdenver.edu

module purge
module load anaconda
conda activate mycustomenv

python param_list_fit_job_5_56.py
