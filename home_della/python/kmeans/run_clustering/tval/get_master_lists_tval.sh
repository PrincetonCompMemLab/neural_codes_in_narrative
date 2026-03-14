#!/usr/bin/env bash
#SBATCH -t 100
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_masters_tval
#SBATCH -c 1
#SBATCH --constraint=cascade
#SBATCH --mem=185G


module load anaconda3/2021.5
conda activate wedding_schema

python get_master_lists_tval.py