#!/usr/bin/env bash
#SBATCH -t 60
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name silhouttes_tval
#SBATCH -c 32
#SBATCH --array=2-3
#SBATCH --mem=350GB
#SBATCH -N 1


module load anaconda3/2021.5
conda activate wedding_schema
python get_s_scores_sklearn_tval.py