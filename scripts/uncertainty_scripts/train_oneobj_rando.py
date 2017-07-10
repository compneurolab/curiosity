'''
A second test for the curious uncertainty loop.
This one's for cluster training, not local.
'''

import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train, environment
from curiosity.interaction.models import another_sample_cfg
from tfutils import base, optimizer
import numpy as np
import os

NUM_BATCHES_PER_EPOCH = 1e8
RENDER2_HOST_ADDRESS = '10.102.2.162'

EXP_ID = 'tlo_1obj_random'
CACHE_ID_PREFIX = '/mnt/fs0/nhaber/cache'
CACHE_DIR = os.path.join(CACHE_ID_PREFIX, EXP_ID)
if not os.path.exists(CACHE_DIR):
	os.mkdir(CACHE_DIR)

STATE_DESC = 'depths1'


another_sample_cfg['uncertainty_model']['state_descriptor'] = STATE_DESC
another_sample_cfg['uncertainty_model']['just_random'] = 0

params = {

	'save_params' : {	
		'host' : 'localhost',
		'port' : 15841,
		'dbname' : 'uncertain_agent',
		'collname' : 'uniform_action',
		'exp_id' : EXP_ID,
		'save_valid_freq' : 2000,
        'save_filters_freq': 100000,
        'cache_filters_freq': 50000,
	'save_metrics_freq' : 1000,
        'save_initial_filters' : False,
	'cache_dir' : CACHE_DIR,
        'save_to_gfs' : ['wm_prediction', 'wm_tv', 'wm_given', 'batch']
	},


	'load_params' : {
		'exp_id' : EXP_ID,
		'load_param_dict' : None
	},



	'what_to_save_params' : {
		'big_save_keys' : ['um_loss', 'wm_loss', 'wm_prediction', 'wm_tv', 'wm_given'],
		'little_save_keys' : ['um_loss', 'wm_loss'],
		'big_save_len' : 50,
		'big_save_freq' : 10000,
		'state_descriptor' : STATE_DESC
	},

	'model_params' : {
		'func' : train.get_default_models,
 		'cfg' : another_sample_cfg,
 		'action_model_desc' : 'uncertainty_model'
	},

	'data_params' : {
		'func' : train.get_default_data_provider,
		'action_limits' : np.array([1., 1.] + [80. for _ in range(6)]),
		'environment_params' : {
			'random_seed' : 1,
			'unity_seed' : 1,
			'room_dims' : (5., 5.),
			'state_memory_len' : {
					'depths1' : 3
				},
			'rescale_dict' : {
					'depths1' : (64, 64)
				},
			'USE_TDW' : True,
			'host_address' : RENDER2_HOST_ADDRESS,
			'message_memory_len' : 2,
			'action_memory_len' : 2,
			'rng_periodicity' : 1
		},
		'scene_list' : [environment.example_scene_info],
		'scene_lengths' : [1024 * 32],
		'capacity' : 5
	},

	'train_params' : {
		'updater_func' : train.get_default_updater
	},



	'optimizer_params' : {
		'world_model' : {
			'func': optimizer.ClipOptimizer,
			'optimizer_class': tf.train.AdamOptimizer,
			'clip': True,
		},
		'uncertainty_model' : {
			'func': optimizer.ClipOptimizer,
			'optimizer_class': tf.train.AdamOptimizer,
			'clip': True,
		}

	},

	'learning_rate_params' : {
		'world_model' : {
			'func': tf.train.exponential_decay,
			'learning_rate': 1e-5,
			'decay_rate': 1.,
			'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
			'staircase': True
		},
		'uncertainty_model' : {
			'func': tf.train.exponential_decay,
			'learning_rate': 1e-5,
			'decay_rate': 1.,
			'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
			'staircase': True
		}
	},


}


if __name__ == '__main__':
#	raise Exception('FIX TFUTILS TRAINING SAVE')
	os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
	train.train_from_params(**params)







