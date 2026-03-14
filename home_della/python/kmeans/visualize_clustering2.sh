#!/usr/bin/env bash
#SBATCH -t 600
#SBATCH --mail-user=rk1593@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_fingerprints2
#SBATCH -c 10
#SBATCH --constraint=cascade
#SBATCH --array=2-20



module load anaconda3/2021.5
conda activate wedding_schema

python visualize_clustering2.py