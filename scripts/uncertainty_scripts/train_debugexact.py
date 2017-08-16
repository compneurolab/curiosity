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
parser.add_argument('--umloss', default = 0, type = int)
parser.add_argument('--momvalue', default = .9, type = float)
parser.add_argument('--useans', default = False, type = bool)


N_ACTION_SAMPLES = 1000
EXP_ID_PREFIX = 'dbgex'
NUM_BATCHES_PER_EPOCH = 1e8
IMAGE_SCALE = (128, 170)
ACTION_DIM = 5

args = vars(parser.parse_args())

wm_cfg = {
	'action_shape' : [2, ACTION_DIM],
	'state_shape' : [2] + list(IMAGE_SCALE) + [3]
}



um_cfg = {
                        'hidden_depth': 1,
                        'hidden' : {
                                1 : {'num_features' : 1, 'activation' : 'identity', 'dropout' : None}
                        },
			'n_action_samples' : 1000,
			'use_ans' : False
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
                        'momentum' : args['momvalue']
                }

        }




#TODO this is really silly, must have kwarg issue in train
get_debugging_updater = lambda models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params: update_step.DebuggingForceMagUpdater(
							models = models, data_provider = data_provider, optimizer_params = optimizer_params,
									learning_rate_params = learning_rate_params, postprocessor = postprocessor, updater_params = updater_params)


def get_force_models(cfg):
	world_model = models.ForceMagSquareWorldModel(cfg['world_model'])
	um = models.SimpleForceUncertaintyModel(cfg['uncertainty_model'], world_model)
	return {'world_model' : world_model, 'uncertainty_model' : um}



train_params = {
	'updater_func' : get_debugging_updater,
	'updater_kwargs' : {
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


for desc in ['oh_my_god', 'ans', 'model_parameters']:
	load_and_save_params['what_to_save_params']['big_save_keys'].append(desc)
	load_and_save_params['save_params']['save_to_gfs'].append(desc)



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


















