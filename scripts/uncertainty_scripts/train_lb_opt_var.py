'''
With a starting point from train_latent_batching.py, a much more modularized generation script.
These all have no object!
'''


import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train, environment, data, cfg_generation
from curiosity.interaction.models import mario_world_model_config
from tfutils import base, optimizer
import numpy as np
import os

#EXP_IDS = dict(((arch, lr, opt), 'actopt_' + str(arch) + str(lr) + str(opt)) for arch in range(4) for lr in range(6) for opt in range(2)
BATCH_SIZE = 32
STATE_DESC = 'depths1'
arch_idx = int(sys.argv[2])
lr_idx = int(sys.argv[3])
opt_idx = int(sys.argv[4])
EXP_ID = 'actoptfix_' + str(arch_idx) + str(lr_idx) + str(opt_idx)

noobj_scene_info = [
        {
        'type' : 'SHAPENET',
        'scale' : .4,
        'mass' : 1.,
        'scale_var' : .01,
        'num_items' : 0,
        }
        ]

wm_cfg_gen_params = [

{
'encode_deets' : {'sizes' : [3, 3, 3, 3], 'strides' : [2, 2, 2, 2], 'nf' : [32, 32, 32, 32]},
'action_deets' : {'nf' : [256]},
'future_deets' : {'nf' : [512]}
},

{
'encode_deets' : {'sizes' : [5, 5], 'strides' : [2, 2], 'nf' : [4, 4]},
'action_deets' : {'nf' : [256]},
'future_deets' : {'nf' : [512]}
},

{
'encode_deets' : {'sizes' : [3, 3, 3, 3], 'strides' : [2, 2, 2, 2], 'nf' : [32, 32, 32, 32]},
'action_deets' : {'nf' : [256, 256]},
'future_deets' : {'nf' : [512]}
},

{
'encode_deets' : {'sizes' : [5, 5, 3], 'strides' : [2, 2, 2], 'nf' : [4, 4, 4]},
'action_deets' : {'nf' : [256, 256]},
'future_deets' : {'nf' : [256]}
}

]

wm_params = wm_cfg_gen_params[arch_idx]

lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
lr = lrs[lr_idx]

opts = [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]
opt = opts[opt_idx]




dp_config = cfg_generation.generate_batching_data_provider(batch_size = BATCH_SIZE, image_scale = (64, 64), scene_info = noobj_scene_info)

save_params_config = cfg_generation.generate_latent_save_params(EXP_ID, location = 'freud', state_desc = STATE_DESC)

um_cfg = cfg_generation.generate_uncertainty_model_cfg(image_shape = (64, 64), state_desc = STATE_DESC, loss_factor = 1/ float(BATCH_SIZE))

wm_cfg= cfg_generation.generate_latent_marioish_world_model_cfg(image_shape = (64, 64), act_loss_factor = 1/float(BATCH_SIZE), act_loss_type = 'one_l2', **wm_params)

print('printing future model!')
print(wm_cfg['future_model'])







model_cfg = cfg_generation.generate_latent_model_cfg(world_cfg = wm_cfg, uncertainty_cfg = um_cfg)

params = cfg_generation.generate_latent_standards(model_cfg = model_cfg, learning_rate = lr, optimizer_class = opt)

params.update(save_params_config)

params['data_params'] = dp_config






if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
	train.train_from_params(**params)








































