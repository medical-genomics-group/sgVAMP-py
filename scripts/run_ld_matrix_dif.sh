#!/bin/bash
#
#SBATCH --job-name=mpi-bayesw
#SBATCH --output=ld.%N.%J.out
#SBATCH --time=01:00:00
#SBATCH --mem=32G # pool for all cores
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user ariadna.villanuevamarijuan@ista.ac.at

# Define and setup VENV - this need to be done only once.
VENV=${HOME}/myVENVs/${DISTRIB_CODENAME}/py3104mpi414


source $VENV/bin/activate

module load python/3.10.6
module load openmpi/4.1.4

srun python3 ld_matrix_dif.py