#!/bin/bash
source ~/.bash_profile
export IND=$SLURM_ARRAY_TASK_ID
cd /home/yamins
python make_tunnel.py
cd /om/user/yamins/src/curiosity/scripts
python -c "import normalopt; normalopt.main($IND)"
