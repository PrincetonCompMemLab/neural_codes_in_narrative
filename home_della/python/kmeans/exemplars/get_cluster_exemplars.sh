#!/usr/bin/env bash
#SBATCH -t 20
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_cluster_exemplars
#SBATCH -c 10
#SBATCH --array=2-10
#SBATCH -N 1


module load anaconda3/2021.5
conda activate wedding_schema

python get_cluster_exemplars.py