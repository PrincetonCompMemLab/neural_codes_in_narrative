#!/usr/bin/env bash
#SBATCH -t 1000
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_fingerprints_plus_bounds_tval_collapseTR
#SBATCH -c 7
#SBATCH --constraint=cascade
#SBATCH --array=5-7
#SBATCH --mem=600GB
#SBATCH -N 1

module load anaconda3/2021.5
conda activate wedding_schema
python visualize_clustering_tval_collapseTR.py