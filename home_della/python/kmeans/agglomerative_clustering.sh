#!/usr/bin/env bash
#SBATCH -t 500
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name agg_and_silhouttes
#SBATCH -c 32
#SBATCH --array=2-20
#SBATCH --mem=400GB
#SBATCH -N 1


module load anaconda3/2021.5
conda activate wedding_schema

python agglomerative_clustering.py