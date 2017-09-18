'''
A script for saving data to an hdf5, so we do some fancy offline learning.
'''



import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf
import argparse
from curiosity.interaction import models, train, environment, data, cfg_generation
from curiosity.interaction.models import mario_world_model_config
from tfutils import base, optimizer
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = '0', type = str)
parser.add_argument('--tasknum', required = True, type = str)
parser.add_argument('--nsave', default = 256 * 1024, type = int)
parser.add_argument('--forcescaling', default = 200., type = float)

args = vars(parser.parse_args())


tasknum = args['tasknum']
batch_size = 32
act_dim = 5
history_len = batch_size
do_torque = False
obj_size = .4
scene_len = 32 * 1024
image_shape = image_scale =  (128, 170)
RENDER1_HOST_ADDRESS = '10.102.2.161'
force_scaling = args['forcescaling']
save_filename = '/media/data2/nhaber/repeat' + str(int(force_scaling)) + '_' + tasknum + '.hdf5'
room_dims = (5., 5.)
state_time_length = 2
data_seed = int(tasknum) + 10
ACTION_REPEAT = 4



scene_info = [
        {
        'type' : 'SHAPENET',
        'scale' : obj_size,
        'mass' : 1.,
        'scale_var' : .01,
        'num_items' : 1,
        }
        ]


my_rng = np.random.RandomState(0)
data_params = {
                'func' : train.get_batching_data_provider,
                'action_limits' : np.array([.25, .25] + [force_scaling for _ in range(act_dim - 2)]),
                'environment_params' : {
                        'random_seed' : 1,
                        'unity_seed' : 1,
                        'room_dims' : room_dims,
                        'state_memory_len' : {
                                        'depths1' : history_len + state_time_length + 1
                                },
                        'action_memory_len' : history_len + state_time_length,
                        'message_memory_len' : batch_size,
                        'other_data_memory_length' : batch_size,
                        'rescale_dict' : {
                                        'depths1' : image_shape
                                },
                        'USE_TDW' : True,
                        'host_address' : RENDER1_HOST_ADDRESS
                },

                'provider_params' : {
                        'batching_fn' : lambda hist : data.uniform_experience_replay(hist, history_len, my_rng = my_rng, batch_size = batch_size),
                        'capacity' : 5,
                        'gather_per_batch' : batch_size,
                        'gather_at_beginning' : history_len + state_time_length + 1,
                	'action_repeat' : 4
		},

                'scene_list' : [scene_info],
                'scene_lengths' : [scene_len],
                'do_torque' : do_torque,
		'use_absolute_coordinates' : False


        }





wm_params = {
'encode_deets' : {'sizes' : [3, 3, 3, 3], 'strides' : [2, 2, 2, 2], 'nf' : [32, 32, 32, 32]},
'action_deets' : {'nf' : [256]},
'future_deets' : {'nf' : [512]}
}




wm_cfg= cfg_generation.generate_latent_marioish_world_model_cfg(image_shape = image_scale, act_loss_type = 'one_l2', include_previous_action = False, action_dim = 5, num_classes = 21, **wm_params)



um_encoding_args = {}
um_mlp_args = {}
um_cfg = {
        'use_world_encoding' : False,
        'encode' : cfg_generation.generate_conv_architecture_cfg(desc = 'encode', **um_encoding_args),
        'mlp' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_args),
        'heat' : 1.,
        'wm_loss' : {
                'func' : models.get_mixed_loss,
                'kwargs' : {
                        'weighting' : {'action' : 1.0, 'future' : 0.0}
                }
        },
        'loss_func' : models.l2_loss,
        'loss_factor' : 1. / float(batch_size),
        'only_model_ego' : False,
        'n_action_samples' : 1000,
	'just_random' : data_seed

}



model_cfg = {
        'world_model' : wm_cfg,
        'uncertainty_model' : um_cfg,
        'seed' : data_seed


}


model_params = {
                'func' : train.get_latent_models,
                'cfg' : model_cfg,
                'action_model_desc' : 'uncertainty_model'
        }



train_params = {
	'updater_kwargs' : {
		'hdf5_filename' : save_filename,
		'N_save' : args['nsave'],
		'image_shape' : image_shape,
		'act_dim' : act_dim
	}
}



params = {
	'model_params' : model_params,
	'data_params' : data_params,
	'train_params' : train_params
}



if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
	train.save_data_without_training(**params)











