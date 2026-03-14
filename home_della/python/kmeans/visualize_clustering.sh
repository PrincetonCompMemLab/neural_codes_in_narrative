#!/usr/bin/env bash
#SBATCH -t 400
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_fingerprints_plus_bounds
#SBATCH -c 5
#SBATCH --constraint=cascade
#SBATCH --array=5
#SBATCH --mem=400GB
#SBATCH -N 1


module load anaconda3/2021.5
conda activate wedding_schema

python visualize_clustering.py