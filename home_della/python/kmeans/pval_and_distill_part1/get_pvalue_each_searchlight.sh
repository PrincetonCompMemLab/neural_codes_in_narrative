#!/usr/bin/env bash
#SBATCH -t 30
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_pvalue_each_searchlight
#SBATCH -c 3
#SBATCH --constraint=cascade
#SBATCH --array=0-199


module load anaconda3/2021.5
conda activate wedding_schema
python get_pvalue_each_searchlight.py