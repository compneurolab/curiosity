'''
A script for training on static data, no interaction.

First iteration, uniform sampling, l2 loss.
'''




import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train, environment, static_data, cfg_generation, update_step
import curiosity.interaction.models as models
from tfutils import base, optimizer
import numpy as np
import os
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = '0', type = str)
parser.add_argument('-ea', '--encarchitecture', default = 0, type = int)
parser.add_argument('-fca', '--fcarchitecture', default = 0, type = int)
parser.add_argument('-mbca', '--mbcarchitecture', default = -1, type = int)
parser.add_argument('--umlr', default = 1e-3, type = float)
parser.add_argument('--actlr', default = 1e-3, type = float)
#parser.add_argument('--loss', default = 0, type = int)
#parser.add_argument('--tiedencoding', default = False, type = bool)
parser.add_argument('--heat', default = 1., type = float)
#parser.add_argument('--egoonly', default = False, type = bool)
#parser.add_argument('--zeroedforce', default = False, type = bool)
parser.add_argument('--optimizer', default = 'adam', type = str)
#parser.add_argument('--batching', default = 'uniform', type = str)
parser.add_argument('--batchsize', default = 32, type = int)
#parser.add_argument('--numperbatch', default = 8, type = int)
#parser.add_argument('--historylen', default = 1000, type = int)
#parser.add_argument('--ratio', default = 2 / .17, type = float)
#parser.add_argument('--objsize', default = .4, type = float)
parser.add_argument('--lossfac', default = 1., type = float)
parser.add_argument('--nclasses', default = 4, type = int)
parser.add_argument('-opb', '--objperbatch', default = 16, type = int)


N_ACTION_SAMPLES = 1000
EXP_ID_PREFIX = 'mi3cl'
NUM_BATCHES_PER_EPOCH = 1e8
IMAGE_SCALE = (128, 170)
ACTION_DIM = 5
NUM_TIMESTEPS = 1
T_PER_STATE = 2
RENDER1_HOST_ADDRESS = '10.102.2.161'
UNIFORM_METADATA_LOC = '/media/data2/nhaber/uniform_5_4.pkl'
OBJTHERE_METADATA_LOC = '/media/data2/nhaber/objthere_5_4.pkl'
STATE_STEPS = [-1, 0]
STATES_GIVEN = [-2, -1, 0, 1]
ACTIONS_GIVEN = [-2, -1, 1]


s_back = - (min(STATES_GIVEN) + min(STATE_STEPS))
s_forward = max(STATES_GIVEN) + max(STATE_STEPS)
a_back = - min(ACTIONS_GIVEN)
a_forward = max(ACTIONS_GIVEN)


args = vars(parser.parse_args())
#act_thresholds = [.1 * k for k in range(-10, 10)]
act_thresholds = [-.1, .1]
n_classes_wm = len(act_thresholds) + 1
n_classes_um = 4


wm_encoding_choices = [
	{
		'sizes' : [3, 3, 3, 3], 
		'strides' : [2, 2, 2, 2], 
		'num_filters' : [32, 32, 32, 32], 
		'bypass' : [None, None, None, None]
	},

	{
                'sizes' : [7, 3, 3, 3],
                'strides' : [3, 2, 2, 2],
                'num_filters' : [32, 32, 32, 32],
                'bypass' : [0, 0, 0, 0]
        },

 	{
                'sizes' : [7, 3, 3, 3, 3],
                'strides' : [3, 2, 2, 2, 2],
                'num_filters' : [32, 32, 32, 32, 32],
                'bypass' : [0, 0, 0, 0, 0]
        }
]

wm_mlp_before_concat_choices = [
	{
                'num_features' : [500, 10],
                'nonlinearities' : ['relu', 'relu']
        },
        {
                'num_features' : [500, 50],
                'nonlinearities' : ['relu', 'relu']
        }




]


wm_mlp_choices = [
	{
		'num_features' : [256, ACTION_DIM * n_classes_wm],
		'nonlinearities' : ['relu', 'identity'],
		'dropout' : [None, None]
	},

	{
		'num_features' : [50, 50, ACTION_DIM * n_classes_wm],
		'nonlinearities' : ['relu', 'relu', 'identity'],
		'dropout' : [None, None, None]
	},

	{
		'num_features' : [50, 50, ACTION_DIM * n_classes_wm],
		'nonlinearities' : [['crelu', 'square_crelu'], ['crelu', 'square_crelu'], 'identity'],
		'dropout' : [None, None, None]
	},

	{
		'num_features' : [100, 100, 100, ACTION_DIM * n_classes_wm],
		'nonlinearities' : [['crelu', 'square_crelu'], ['crelu', 'square_crelu'], ['crelu', 'square_crelu'], 'identity'],
		'dropout' : [None, None, None, None]
	},

        {
                'num_features' : [500, 500, ACTION_DIM * n_classes_wm],
                'nonlinearities' : [['crelu', 'square_crelu'], ['crelu', 'square_crelu'], 'identity'],
                'dropout' : [None, None, None]
        },

	{
                'num_features' : [1000, 1000, 500, ACTION_DIM * n_classes_wm],
                'nonlinearities' : [['crelu', 'square_crelu'], ['crelu', 'square_crelu'], ['crelu', 'square_crelu'], 'identity'],
                'dropout' : [None, None, None, None]
	}


]


wm_encoding_choice = wm_encoding_choices[args['encarchitecture']]
wm_mlp_choice = wm_mlp_choices[args['fcarchitecture']]


wm_cfg = {
	'num_timesteps' : NUM_TIMESTEPS,
	'state_steps' : [-1, 0],
	'image_shape' : list(IMAGE_SCALE) + [3],
	'states_given' : [-2, -1, 0, 1],
	'actions_given' : [-2, -1, 1],
	'act_dim' : ACTION_DIM,
	'encode' : cfg_generation.generate_conv_architecture_cfg(**wm_encoding_choice),
	'action_model' : {
		'loss_func' : models.binned_softmax_loss_per_example,
		'thresholds': act_thresholds,
		'loss_factor' : 1.,
		'mlp' : cfg_generation.generate_mlp_architecture_cfg(**wm_mlp_choice)
	}
}


mbc_idx = args['mbcarchitecture']
if mbc_idx != -1:
	wm_mbc_choice = wm_mlp_before_concat_choices[mbc_idx]
	wm_cfg['action_model']['mlp_before_concat'] = cfg_generation.generate_mlp_architecture_cfg(**wm_mbc_choice)

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
		'num_features' : [n_classes_um],
		'nonlinearities' : ['identity'],
		'dropout' : [None]
	}

separate_mlp_choice = dict((t, separate_mlp_choices_proto) for t in range(NUM_TIMESTEPS))



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



#just hard-coding this! not the point...
um_encoding_args = um_encoding_choices[0]
um_mlp_before_act_args = mlp_before_action_choices[1]
um_mlp_args = shared_mlp_choices[2]


um_cfg = {
	'shared_encode' : cfg_generation.generate_conv_architecture_cfg(desc = 'encode', **um_encoding_args),
	'shared_mlp_before_action' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_before_act_args),
	'shared_mlp' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_args),
	'mlp' : dict((t, cfg_generation.generate_mlp_architecture_cfg(**choice_args)) for t, choice_args in separate_mlp_choice.iteritems()),
	'loss_func' : models.ms_sum_binned_softmax_loss,
	'thresholds' : [.05, .3, .6],
	'loss_factor' : args['lossfac'],
	'n_action_samples' : N_ACTION_SAMPLES,
	'heat' : args['heat']
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
	'updater_func' : update_step.ActionUncertaintyUpdater,
	'updater_kwargs' : {
		'state_desc' : 'depths1'

	}
}


def get_ms_models(cfg):
	world_model = models.MoreInfoActionWorldModel(cfg['world_model'])
	uncertainty_model = models.MSExpectedUncertaintyModel(cfg['uncertainty_model'], world_model)
	return {'world_model' : world_model, 'uncertainty_model' : uncertainty_model}

model_params = {
                'func' : get_ms_models,
                'cfg' : model_cfg,
                'action_model_desc' : 'uncertainty_model'
        }



data_lengths = {
			'obs' : {'depths1' : s_back + s_forward + NUM_TIMESTEPS}, 
			'action' : a_back + a_forward + NUM_TIMESTEPS, 
			'action_post' : a_back + a_forward + NUM_TIMESTEPS}



def get_static_data_provider(data_params, model_params, action_model):
	data_params_copy = copy.copy(data_params)
	data_params_copy.pop('func')
        return static_data.OfflineDataProvider(**data_params_copy)



num_there_per_batch = args['objperbatch']
assert num_there_per_batch <= 32 and num_there_per_batch >= 0

dp_config = {
                'func' : get_static_data_provider,
                'batch_size' : args['batchsize'],
		'batcher_constructor' : static_data.ObjectThereBatcher,
		'data_lengths' : data_lengths,
		'capacity' : 5,
        	'metadata_filename' : OBJTHERE_METADATA_LOC,
		'batcher_kwargs' : {
                	'seed' : 0,
			'num_there_per_batch' : num_there_per_batch,
			'num_not_there_per_batch' : args['batchsize'] - num_there_per_batch
        	}	
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


















