'''
A script for searching through various possibilities with uncertainty model online learning, assuming 
that we are modeling the uncertainty of the action model.
'''




import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train, environment, data, cfg_generation, update_step
import curiosity.interaction.models as models
from tfutils import base, optimizer
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
#architecture specifics. need to make this setup exactly like in train_
parser.add_argument('-g', '--gpu', default = '0', type = str)
parser.add_argument('-ea', '--encarchitecture', default = 0, type = int)
parser.add_argument('-fca', '--fcarchitecture', default = 0, type = int)
#parser.add_argument('--umlr', default = 1e-3, type = float)
#parser.add_argument('--actlr', default = 1e-3, type = float)
#parser.add_argument('--loss', default = 0, type = int)
#parser.add_argument('--tiedencoding', default = False, type = bool)
#parser.add_argument('--heat', default = 1., type = float)
#parser.add_argument('--egoonly', default = False, type = bool)
#parser.add_argument('--zeroedforce', default = False, type = bool)
#parser.add_argument('--optimizer', default = 'adam', type = str)

#data provider details. numperbatch should be high, clearing out the memory and allowing for a fairly uncorrelated test.
parser.add_argument('--batching', default = 'uniform', type = str)
parser.add_argument('--batchsize', default = 32, type = int)
parser.add_argument('--numperbatch', default = 500, type = int)
parser.add_argument('--historylen', default = 500, type = int)
parser.add_argument('--ratio', default = 2 / .17, type = float)
parser.add_argument('--objsize', default = .4, type = float)
parser.add_argument('--numsteps', default = 20, type = int)

N_ACTION_SAMPLES = 1000
EXP_ID_PREFIX = 'objthere'
NUM_BATCHES_PER_EPOCH = 1e8
IMAGE_SCALE = (128, 170)
ACTION_DIM = 5
FORCE_SCALING = 80.
RENDER1_HOST_ADDRESS = '10.102.2.161'
scene_len = 250


args = vars(parser.parse_args())


#model configuration
wm_cfg = {
	'action_shape' : [2, ACTION_DIM],
	'state_shape' : [2] + list(IMAGE_SCALE) + [3]
}



um_encoding_choices = [
	{}, #default
	{
		'sizes' : [3, 3, 3, 3],
		'strides' : [2, 2, 2, 2],
		'num_filters' : [32, 32, 16, 8],
		'bypass' : [None, None, None, None]
	},
	{
		'sizes' : [5, 5, 5],
		'strides' : [2, 2, 2],
		'num_filters' : [6, 6, 6],
		'bypass' : [None, None, None],
	},
	{
		'sizes' : [3, 3, 3, 3],
		'strides' : [2, 2, 2, 2],
		'num_filters' : [32, 32, 32, 32],
		'bypass' : [0, 0, 0, 0]
	},
	{
		'sizes' : [7, 3, 3, 3],
		'strides' : [3, 2, 2, 2],
		'num_filters' : [32, 32, 32, 32],
		'bypass' : [0, 0, 0, 0]
	}

]







um_mlp_choices = [
	{
		'num_features' : [100, 2],
		'nonlinearities' : ['relu', 'identity']
	},
	{
		'num_features' : [50, 50, 2],
		'nonlinearities' : ['relu', 'relu', 'identity'],
		'dropout' : [None, None, None]
	},
	{
		'num_features' : [500, 2],
		'nonlinearities' : ['relu', 'identity']
	},

	{
		'num_features': [2],
		'nonlinearities' : ['identity'],
		'dropout' : [None]

	}


]




um_encoding_args = um_encoding_choices[args['encarchitecture']]
um_mlp_args = um_mlp_choices[args['fcarchitecture']]


um_cfg = {
	'use_world_encoding' : False,
	'encode' : cfg_generation.generate_conv_architecture_cfg(desc = 'encode', **um_encoding_args), 
	'mlp' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_args),
	'heat' : 1.,
	'wm_loss' : {
		'func' : models.get_obj_there,
		'kwargs' : {}
	},
	'loss_func' : models.categorical_loss,
	'loss_factor' : 1.,
	'only_model_ego' : False,
	'n_action_samples' : N_ACTION_SAMPLES
}



model_cfg = {
	'world_model' : wm_cfg,
	'uncertainty_model' : um_cfg,
	'seed' : 0


}


def get_objthere_models(cfg):
	world_model = models.ObjectThereWorldModel(cfg['world_model'])
	um = models.UncertaintyModel(cfg['uncertainty_model'], world_model)
	return {'world_model' : world_model, 'uncertainty_model' : um}



model_params = {
                'func' : get_objthere_models,
                'cfg' : model_cfg,
                'action_model_desc' : 'uncertainty_model'
        }



validate_params = {
	'func' : update_step.ObjectThereValidater,
	'kwargs' : {},
	'num_steps' : args['numsteps']
}




scene_info = [
        {
        'type' : 'SHAPENET',
        'scale' : args['objsize'],
        'mass' : 1.,
        'scale_var' : .01,
        'num_items' : 1,
        }
        ]


history_len = args['historylen']
state_time_length = 2
num_gathered_per_batch = args['numperbatch']
my_rng = np.random.RandomState(2)
batch_size = args['batchsize']
ratio = args['ratio']
which_batching = args['batching']
act_dim = ACTION_DIM

if which_batching == 'uniform':
	dp_config = {
                'func' : train.get_batching_data_provider,
                'action_limits' : np.array([1., 1.] + [FORCE_SCALING for _ in range(act_dim - 2)]),
                'environment_params' : {
                        'random_seed' : 2,
                        'unity_seed' : 2,
                        'room_dims' : (5, 5),
                        'state_memory_len' : {
                                        'depths1' : history_len + state_time_length + 1
                                },
                        'action_memory_len' : history_len + state_time_length,
                        'message_memory_len' : history_len + state_time_length,
                        'other_data_memory_length' : 32,
                        'rescale_dict' : {
                                        'depths1' : IMAGE_SCALE
                                },
                        'USE_TDW' : True,
                        'host_address' : RENDER1_HOST_ADDRESS
                },

                'provider_params' : {
                        'batching_fn' : lambda hist : data.uniform_experience_replay(hist, history_len, my_rng = my_rng, batch_size = batch_size,
                                        get_object_there_binary = True),
                        'capacity' : 5,
                        'gather_per_batch' : num_gathered_per_batch,
                        'gather_at_beginning' : history_len + state_time_length + 1
                },

                'scene_list' : [scene_info],
                'scene_lengths' : [scene_len],
                'do_torque' : False




        }
elif which_batching == 'obj_there':
        act_dim = 8 if do_torque else 5
        if num_gathered_per_batch is None:
                num_gathered_per_batch = batch_size / 4
        dp_config = {
                'func' : train.get_batching_data_provider,
                'action_limits' : np.array([1., 1.] + [FORCE_SCALING for _ in range(act_dim - 2)]),
                'environment_params' : {
                        'random_seed' : 2,
                        'unity_seed' : 2,
                        'room_dims' : (5, 5),
                        'state_memory_len' : {
                                        'depths1' : history_len + state_time_length + 1
                                },
                        'action_memory_len' : history_len + state_time_length,
                        'message_memory_len' : history_len + state_time_length,
                        'other_data_memory_length' : 32,
                        'rescale_dict' : {
                                        'depths1' : IMAGE_SCALE
                                },
                        'USE_TDW' : True,
                        'host_address' : RENDER1_HOST_ADDRESS
                },

                'provider_params' : {
                        'batching_fn' : lambda hist : data.obj_there_experience_replay(hist, history_len, my_rng = my_rng, batch_size = batch_size, there_not_there_ratio = ratio, get_object_there_binary = True),
                        'capacity' : 5,
                        'gather_per_batch' : num_gathered_per_batch,
                        'gather_at_beginning' : history_len + state_time_length + 1
                },

                'scene_list' : [scene_info],
                'scene_lengths' : [scene_len],
                'do_torque' : False
        }
else:
	raise Exception('Data provider option not reconized')







load_and_save_params = cfg_generation.query_gen_latent_save_params(location = 'freud', prefix = EXP_ID_PREFIX, state_desc = 'depths1', load_and_save_elsewhere = True)


load_and_save_params.pop('what_to_save_params')


params = {
	'model_params' : model_params,
	'data_params' : dp_config,
	'validate_params' : validate_params
}

params.update(load_and_save_params)








if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
	train.test_from_params(**params)


















