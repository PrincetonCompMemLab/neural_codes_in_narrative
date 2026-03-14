#!/usr/bin/env bash
#SBATCH -t 100
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name run_jobs_0_through_200_tval
#SBATCH -c 32
#SBATCH --constraint=cascade
#SBATCH --array=0-200
#SBATCH --mem=185G


module load anaconda3/2021.5
conda activate wedding_schema

python run_job_tval.py