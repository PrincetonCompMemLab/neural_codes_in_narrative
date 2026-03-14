#!/usr/bin/env bash
#SBATCH -t 60
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name kmeans_tval
#SBATCH -c 32
#SBATCH --constraint=cascade
#SBATCH --array=2-3



module load anaconda3/2021.5
conda activate wedding_schema

python kmeans_clustering_tval.py