#!/bin/bash
#SBATCH -n 1
#SBATCH --error=/om/user/yamins/src/curiosity/scripts/normalopt_logs/%a.error
#SBATCH --output=/om/user/yamins/src/curiousity/scripts/normalopt_logs/%a.out
srun -N1 /om/user/yamins/src/curiosity/scripts/normalopt.sh
