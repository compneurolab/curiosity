'''
A script for searching through various possibilities with uncertainty model online learning, assuming 
that we are modeling the uncertainty of the action model.
'''




import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train, environment, data, cfg_generation
import curiosity.interaction.models as models
from tfutils import base, optimizer
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = '0', type = str)
parser.add_argument('-ea', '--encarchitecture', default = 0, type = int)
parser.add_argument('-fca', '--fcarchitecture', default = 0, type = int)
parser.add_argument('-mbaa', '--mbaarchitecture', default = 0, type = int)
parser.add_argument('--umlr', default = 1e-3, type = float)
parser.add_argument('--actlr', default = 1e-3, type = float)
#parser.add_argument('--loss', default = 0, type = int)
parser.add_argument('--tiedencoding', default = False, type = bool)
parser.add_argument('--heat', default = 1., type = float)
parser.add_argument('--egoonly', default = False, type = bool)
parser.add_argument('--zeroedforce', default = False, type = bool)
parser.add_argument('--optimizer', default = 'adam', type = str)
parser.add_argument('--batching', default = 'uniform', type = str)
parser.add_argument('--batchsize', default = 32, type = int)
parser.add_argument('--numperbatch', default = 8, type = int)
parser.add_argument('--historylen', default = 1000, type = int)
parser.add_argument('--ratio', default = 2 / .17, type = float)
parser.add_argument('--objsize', default = .4, type = float)
parser.add_argument('--lossfac', default = 1., type = float)
parser.add_argument('--nclasses', default = 4, type = int)


N_ACTION_SAMPLES = 1000
EXP_ID_PREFIX = 'cs'
NUM_BATCHES_PER_EPOCH = 1e8
IMAGE_SCALE = (128, 170)
ACTION_DIM = 5
NUM_TIMESTEPS = 3
T_PER_STATE = 2
RENDER1_HOST_ADDRESS = '10.102.2.161'

args = vars(parser.parse_args())

n_classes = args['nclasses']


wm_cfg = {
	'num_timesteps' : 3,
	'state_shape' : [T_PER_STATE] + list(IMAGE_SHAPE) + [3],
	'act_dim' = ACTION_DIM,
	'encode' : cfg_generation.generate_conv_architecture_cfg({'sizes' : [3, 3, 3, 3], 'strides' : [2, 2, 2, 2], 'num_filters' : [32, 32, 32, 32]})
	'action_model' : {
		'mlp' : cfg_generation.generate_mlp_architecture_cfg({'num_features' : [256, ACTION_DIM])
	}
}



um_encoding_choices = [

        {
                'sizes' : [7, 3, 3, 3],
                'strides' : [3, 2, 2, 2],
                'num_filters' : [32, 32, 32, 32],
                'bypass' : [0, 0, 0, 0]
        },

	{
		'sizes' : [7, 3],
		'strides' : [3, 2],
		'num_filters' : [16, 2],
		'bypass' : [0, 0]
	},

	{
		'sizes' : [7, 3, 3, 3, 3],
		'strides' : [3, 2, 2, 2, 2],
		'num_filters' : [32, 32, 32, 32, 32],
		'bypass' : [0, 0, 0, 0, 0]
	}

]



shared_mlp_choices = [
	{
		'num_features' : [100, 100],
		'nonlinearities' : ['relu', 'relu'],
		'dropout' : [None, None]
	},

	{
		'num_features' : [50, 50],
		'nonlinearities' : ['relu', 'relu'],
		'dropout' : [None, None]
	},

	{
		'num_features' : [500],
		'nonlinearities' : ['relu'],
		'dropout' : [None]
	},

	{
		'num_features' : [50, 50],
		'nonlinearities' : [['crelu', 'square_crelu'], ['crelu', 'square_crelu']],
		'dropout' : [None, None]
	}
]



separate_mlp_choices_proto = {
		'num_features' : [n_classes],
		'nonlinearities' : ['identity']
		'dropout' : [None]
	}

separate_mlp_choice = dict((t, separate_mlp_choices_proto) for t in NUM_TIMESTEPS)



mlp_before_action_choices = [
	{
		'num_features' : [500, 10],
		'nonlinearities' : ['relu', 'relu']
	},
	{
		'num_features' : [500, 50],
		'nonlinearities' : ['relu', 'relu']
	},
	{
		'num_features' : [300, 100],
		'nonlinearities' : ['relu', 'relu']
	}
]




um_encoding_args = um_encoding_choices[args['encarchitecture']]
um_mlp_before_act_args = mlp_before_action_choices[args['mbaarchitecture']]
um_mlp_args = shared_mlp_choices[args['fcarchitecture']]


um_cfg = {
	'shared_encode' : cfg_generation.generate_conv_architecture_cfg(desc = 'encode', **um_encoding_args),
	'shared_mlp_before_action' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_before_act_args),
	'shared_mlp' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_args),
	'mlp' : dict((t, cfg_generation.generate_mlp_architecture_cfg(**choice_args)) for t, choice_args in separate_mlp_choice.iteritems())
	'loss_func' : models.ms_sum_binned_softmax_loss,
	'thresholds' : [.05, .3, .6],
	'loss_factor' : args['lossfac'],
	'n_action_samples' : N_ACTION_SAMPLES,
	'just_random' : 1
}

model_cfg = {
	'world_model' : wm_cfg,
	'uncertainty_model' : um_cfg,
	'seed' : 0


}


lr_params = {              
		'world_model' : {
                        'act_model' : {
                        'func': tf.train.exponential_decay,
                        'learning_rate': args['actlr'],
                        'decay_rate': 1.,
                        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                        'staircase': True
                        },
                        'fut_model' : {
                        'func': tf.train.exponential_decay,
                        'learning_rate': args['actlr'],
                        'decay_rate': 1.,
                        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                        'staircase': True
                }
                },
                'uncertainty_model' : {
                        'func': tf.train.exponential_decay,
                        'learning_rate': args['umlr'],
                        'decay_rate': 1.,
                        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                        'staircase': True
                }
}



if args['optimizer'] == 'adam':
	optimizer_class = tf.train.AdamOptimizer
	optimizer_params = {
                'world_model' : {
                        'act_model' : {
                                'func': optimizer.ClipOptimizer,
                                'optimizer_class': optimizer_class,
                                'clip': True,
                        },
                        'fut_model' : {
                                'func': optimizer.ClipOptimizer,
                                'optimizer_class': optimizer_class,
                                'clip': True,
                }
                },
                'uncertainty_model' : {
                        'func': optimizer.ClipOptimizer,
                        'optimizer_class': optimizer_class,
                        'clip': True,
                }

        }
elif args['optimizer'] == 'momentum':
	optimizer_class = tf.train.MomentumOptimizer
	optimizer_params = {
                'world_model' : {
                        'act_model' : {
                                'func': optimizer.ClipOptimizer,
                                'optimizer_class': optimizer_class,
                                'clip': True,
                                'momentum' : .9
                        },
                        'fut_model' : {
                                'func': optimizer.ClipOptimizer,
                                'optimizer_class': optimizer_class,
                                'clip': True,
                                'momentum' : .9
                }
                },
                'uncertainty_model' : {
                        'func': optimizer.ClipOptimizer,
                        'optimizer_class': optimizer_class,
                        'clip': True,
                        'momentum' : .9
                }

        }



train_params = {
	'updater_func' : ActionUncertaintyUpdater,
	'updater_kwargs' : {
		'state_desc' : 'depths1'

	}
}


def get_ms_models(cfg):
	world_model = MSActionWorldModel(cfg['world_model'])
	uncertainty_model = MSExpectedUncertaintyModel(cfg['uncertainty_model'], world_model)
	return {'world_model' : world_model, 'uncertainty_model' : uncertainty_model}

model_params = {
                'func' : get_ms_models,
                'cfg' : model_cfg,
                'action_model_desc' : 'uncertainty_model'
        }



one_obj_scene_info = [
        {
        'type' : 'SHAPENET',
        'scale' : args['objsize'],
        'mass' : 1.,
        'scale_var' : .01,
        'num_items' : 1,
        }
        ]


force_scaling = 80.
room_dims = (5, 5)
my_rng = np.random.RandomState(0)
history_len = args['historylen']
batch_size = args['batchsize']

data_lengths = {
			'obs' : {'depths1' : T_PER_STATE + NUM_TIMESTEPS}, 
			'action' : T_PER_STATE - 1 + NUM_TIMESTEPS, 
			'action_post' : T_PER_STATE - 1 + NUM_TIMESTEPS}


dp_config = {
                'func' : train.get_batching_data_provider,
                'action_limits' : np.array([1., 1.] + [force_scaling for _ in range(act_dim - 2)]),
                'environment_params' : {
                        'random_seed' : 1,
                        'unity_seed' : 1,
                        'room_dims' : room_dims,
                        'state_memory_len' : {
                                        'depths1' : history_len + T_PER_STATE + NUM_TIMESTEPS
                                },
                        'action_memory_len' : history_len + T_PER_STATE + NUM_TIMESTEPS - 1,
                        'message_memory_len' : history_len + T_PER_STATE + NUM_TIMESTEPS - 1,
                        'other_data_memory_length' : 32,
                        'rescale_dict' : {
                                        'depths1' : IMAGE_SCALE
                                },
                        'USE_TDW' : True,
                        'host_address' : RENDER1_HOST_ADDRESS
                },

                'provider_params' : {
                        'batching_fn' : lambda hist : data.uniform_experience_replay(hist, history_len, my_rng = my_rng, batch_size = batch_size,
                                        get_object_there_binary = False, data_lengths = data_lengths),
                        'capacity' : 5,
                        'gather_per_batch' : batch_size / 4,
                        'gather_at_beginning' : history_len + T_PER_STATE + NUM_TIMESTEPS
                },

                'scene_list' : [one_obj_scene_info],
                'scene_lengths' : [1024 * 32],
                'do_torque' : False




        }



load_and_save_params = cfg_generation.query_gen_latent_save_params(location = 'freud', prefix = EXP_ID_PREFIX, state_desc = 'depths1')

postprocessor_params = {
        'func' : train.get_experience_replay_postprocessor

}



params = {
	'model_params' : model_params,
	'data_params' : dp_config,
	'postprocessor_params' : postprocessor_params,
	'optimizer_params' : optimizer_params,
	'learning_rate_params' : lr_params,
	'train_params' : train_params
}

params.update(load_and_save_params)


params['allow_growth'] = True







if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
	train.train_from_params(**params)


















