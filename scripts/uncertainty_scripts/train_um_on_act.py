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
parser.add_argument('--umlr', default = 1e-3, type = float)
parser.add_argument('--actlr', default = 1e-3, type = float)
parser.add_argument('--loss', default = 0, type = int)
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

N_ACTION_SAMPLES = 1000
EXP_ID_PREFIX = 'umtries'
NUM_BATCHES_PER_EPOCH = 1e8
IMAGE_SCALE = (64, 64)
ACTION_DIM = 5

args = vars(parser.parse_args())

wm_arch_params = {
'encode_deets' : {'sizes' : [3, 3, 3, 3], 'strides' : [2, 2, 2, 2], 'nf' : [32, 32, 32, 32]},
'action_deets' : {'nf' : [256]},
'future_deets' : {'nf' : [512]}
}

wm_cfg= cfg_generation.generate_latent_marioish_world_model_cfg(image_shape = IMAGE_SCALE, act_loss_type = 'one_l2', include_previous_action = False, action_dim = ACTION_DIM, **wm_arch_params)





um_encoding_choices = [
	{}, #default
	{
		'sizes' : [3, 3, 3, 3],
		'strides' : [2, 2, 2, 2],
		'num_filters' : [32, 32, 32, 32],
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
	}

]







um_mlp_choices = [
	{}, #default
	{
		'num_features' : [100, 1],
		'nonlinearities' : ['relu', 'identity']
	},
	{
		'num_features' : [50, 50, 1],
		'nonlinearities' : ['relu', 'relu', 'identity'],
		'dropout' : [None, None, None]
	},
	{
		'num_features' : [500, 1],
		'nonlinearities' : ['relu', 'identity']
	},


]



um_loss_choices = [
	models.l2_loss,	



]




um_encoding_args = um_encoding_choices[args['encarchitecture']]
um_mlp_args = um_mlp_choices[args['fcarchitecture']]


um_cfg = {
	'use_world_encoding' : args['tiedencoding'],
	'encode' : cfg_generation.generate_conv_architecture_cfg(desc = 'encode', **um_encoding_args), 
	'mlp' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_args),
	'heat' : args['heat'],
	'wm_loss' : {
		'func' : models.get_mixed_loss,
		'kwargs' : {
			'weighting' : {'action' : 1.0, 'future' : 0.0}
		}
	},
	'loss_func' : um_loss_choices[args['loss']],
	'loss_factor' : 1. / float(args['batchsize']),
	'only_model_ego' : args['egoonly'],
	'n_action_samples' : N_ACTION_SAMPLES
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
	'updater_func' : train.get_latent_updater,
	'updater_kwargs' : {
		'state_desc' : 'depths1'

	}
}


model_params = {
                'func' : train.get_latent_models,
                'cfg' : model_cfg,
                'action_model_desc' : 'uncertainty_model'
        }



one_obj_scene_info = [
        {
        'type' : 'SHAPENET',
        'scale' : .4,
        'mass' : 1.,
        'scale_var' : .01,
        'num_items' : 1,
        }
        ]



which_batching = args['batching']

if which_batching == 'uniform':
        dp_config = cfg_generation.generate_experience_replay_data_provider(batch_size = args['batchsize'], image_scale = IMAGE_SCALE, scene_info = one_obj_scene_info, history_len = args['historylen'], do_torque = False)
elif which_batching == 'objthere':
        dp_config = cfg_generation.generate_object_there_experience_replay_provider(batch_size = args['batchsize'], image_scale = IMAGE_SCALE, scene_info = one_obj_scene_info, history_len = args['historylen'], do_torque = False, ratio = args['ratio'], num_gathered_per_batch = args['numperbatch'])
else:
	raise Exception('Invalid batching argument')



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


















