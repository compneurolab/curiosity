#!/bin/bash
source ~/.bash_profile
export IND=$SLURM_ARRAY_TASK_ID
cd /home/yamins
python make_tunnel.py
cd /om/user/yamins/src/curiosity/scripts
python -c "import normalopt; normalopt.main($IND, srcdir='/om/user/yamins/src', savedir='/om/user/yamins/tensorflow_checkpoint_cache', script='normal_encoder_opt_source3.py')"
