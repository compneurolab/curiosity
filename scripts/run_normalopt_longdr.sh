#!/bin/bash
#SBATCH -n 1
#SBATCH --error=/om/user/yamins/src/curiosity/scripts/normalopt_logs/longdr_%a.error
#SBATCH --output=/om/user/yamins/src/curiosity/scripts/normalopt_logs/longdr_%a.out
srun -N1 /om/user/yamins/src/curiosity/scripts/normalopt_longdr.sh
