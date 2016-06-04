#!/bin/bash
source ~/.bash_profile
export IND=$SLURM_ARRAY_TASK_ID
cd /home/yamins
python make_tunnel.py
cd /om/user/yamins/src/curiosity/scripts
python -c "import normalopt; normalopt.main($IND, dbname='normal_encoder_opt1', colname='optimization_0', srcdir='/om/user/yamins/src', savedir='/om/user/yamins/tensorflow_checkpoint_cache', decaystep=1000000)"
