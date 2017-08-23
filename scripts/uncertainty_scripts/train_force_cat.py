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
parser.add_argument('-g', '--gpu', default = '0', type = str)
parser.add_argument('-ea', '--encarchitecture', default = 0, type = int)
parser.add_argument('-fca', '--fcarchitecture', default = 0, type = int)
parser.add_argument('-mbaa', '--mbaarchitecture', default = 0, type = int)
parser.add_argument('--umlr', default = 1e-3, type = float)
#parser.add_argument('--actlr', default = 1e-3, type = float)
#parser.add_argument('--loss', default = 0, type = int)
#parser.add_argument('--tiedencoding', default = False, type = bool)
#parser.add_argument('--heat', default = 1., type = float)
#parser.add_argument('--egoonly', default = False, type = bool)
#parser.add_argument('--zeroedforce', default = False, type = bool)
parser.add_argument('--optimizer', default = 'adam', type = str)
parser.add_argument('--batching', default = 'uniform', type = str)
parser.add_argument('--batchsize', default = 32, type = int)
parser.add_argument('--numperbatch', default = 8, type = int)
parser.add_argument('--historylen', default = 1000, type = int)
parser.add_argument('--ratio', default = 2 / .17, type = float)
parser.add_argument('--objsize', default = .4, type = float)

N_ACTION_SAMPLES = 1000
EXP_ID_PREFIX = 'ct'
NUM_BATCHES_PER_EPOCH = 1e8
IMAGE_SCALE = (128, 170)
ACTION_DIM = 5


args = vars(parser.parse_args())

n_classes = 4

wm_cfg = {
	'action_shape' : [2, ACTION_DIM],
	'state_shape' : [2] + list(IMAGE_SCALE) + [3]
}


um_encoding_choices = [
	{
		'sizes' : [7, 3, 3, 3],
		'strides' : [3, 2, 2, 2],
		'num_filters' : [32, 32, 32, 32],
		'bypass' : [0, 0, 0, 0]
	}

]



um_mlp_choices = [
        {
                'num_features' : [500, n_classes],
                'nonlinearities' : ['relu', 'identity'],
                'dropout' : [None, None]
        },

        {
                'num_features' : [100, 100, n_classes],
                'nonlinearities' : ['relu', 'relu', 'identity'],
                'dropout' : [None, None, None]
        },

        {
                'num_features' : [250, 250, n_classes],
                'nonlinearities' : ['relu', 'relu', 'identity'],
                'dropout' : [None, None, None]
        },

	{
		'num_features' : [50, 50, n_classes],
		'nonlinearities' : ['relu', 'relu', 'identity'],
		'dropout' : [None, None, None]
	},

	{
		'num_features' : [50, 50, 50, n_classes],
		'nonlinearities' : ['relu', 'relu', 'relu', 'identity'],
		'dropout' : [None, None, None, None]
	},



]







mlp_before_action_choices = [
        {
                'num_features' : [500, 1],
                'nonlinearities' : ['relu', 'relu']
        },
        {
                'num_features' : [500, 1],
                'nonlinearities' : ['relu', 'tanh']
        }
]





um_encoding_args = um_encoding_choices[args['encarchitecture']]
um_mlp_args = um_mlp_choices[args['fcarchitecture']]
um_mlp_before_act_args = mlp_before_action_choices[args['mbaarchitecture']]


um_cfg = {
	'use_world_encoding' : False,
	'encode' : cfg_generation.generate_conv_architecture_cfg(desc = 'encode', **um_encoding_args), 
        'mlp_before_action' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_before_act_args),
	'mlp' : cfg_generation.generate_mlp_architecture_cfg(**um_mlp_args),
	'heat' : 1.,
	'wm_loss' : {
		'func' : models.get_force_square,
		'kwargs' : {}
	},
	'just_random' : 1,
	'loss_func' : models.binned_softmax_loss,
	'thresholds' : [.0001, .3, .6],
	'only_model_ego' : False,
	'loss_factor' : 1., 
	'n_action_samples' : N_ACTION_SAMPLES
}



model_cfg = {
	'world_model' : wm_cfg,
	'uncertainty_model' : um_cfg,
	'seed' : 0


}


lr_params = {              
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
              'uncertainty_model' : {
                        'func': optimizer.ClipOptimizer,
                        'optimizer_class': optimizer_class,
                        'clip': True,
                }

        }
elif args['optimizer'] == 'momentum':
	optimizer_class = tf.train.MomentumOptimizer
	optimizer_params = {
                'uncertainty_model' : {
                        'func': optimizer.ClipOptimizer,
                        'optimizer_class': optimizer_class,
                        'clip': True,
                        'momentum' : .9
                }

        }


get_force_updater = lambda models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params : \
                                update_step.SquareForceMagUpdater(models = models, data_provider = data_provider, optimizer_params = optimizer_params,
                                                learning_rate_params = learning_rate_params, postprocessor = postprocessor,
                                                updater_params = updater_params)



def get_force_models(cfg):
	world_model = models.ForceMagSquareWorldModel(cfg['world_model'])
	um = models.UncertaintyModel(cfg['uncertainty_model'], world_model)
	return {'world_model' : world_model, 'uncertainty_model' : um}



train_params = {
	'updater_func' : get_force_updater,
	'updater_kwargs' : {
		'state_desc' : 'depths1'

	}
}





model_params = {
                'func' : get_force_models,
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



which_batching = args['batching']

if which_batching == 'uniform':
        dp_config = cfg_generation.generate_experience_replay_data_provider(batch_size = args['batchsize'], image_scale = IMAGE_SCALE, scene_info = one_obj_scene_info, history_len = args['historylen'], do_torque = False, get_object_there_binary = True)
elif which_batching == 'objthere':
        dp_config = cfg_generation.generate_object_there_experience_replay_provider(batch_size = args['batchsize'], image_scale = IMAGE_SCALE, scene_info = one_obj_scene_info, history_len = args['historylen'], do_torque = False, ratio = args['ratio'], num_gathered_per_batch = args['numperbatch'], get_object_there_binary = True)
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


















