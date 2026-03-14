#!/usr/bin/env bash
#SBATCH -t 750
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_fingerprints_plus_bounds_tval_perS0.5
#SBATCH -c 1
#SBATCH --constraint=cascade
#SBATCH --array=5
#SBATCH --mem=600GB
#SBATCH -N 1

module load anaconda3/2021.5
conda activate wedding_schema
python visualize_clustering_tval_perS0.5.py