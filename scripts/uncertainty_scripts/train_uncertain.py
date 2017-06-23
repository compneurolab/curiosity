'''
A second test for the curious uncertainty loop.
This one's for cluster training, not local.
'''

import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train
from curiosity.interaction.models import another_sample_cfg
from tfutils import base, optimizer
import numpy as np

NUM_BATCHES_PER_EPOCH = 1e8


params = {

	'save_params' : {	
		'host' : 'localhost',
		'port' : 27017,
		'dbname' : 'uncertain_agent',
		'collname' : 'uniform_action',
#		'exp_id' : 'jerk_corr',
		'save_valid_freq' : 2000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 2000,
        'save_initial_filters' : False,
#        'cache_dir' : CACHE_DIR,
        'save_to_gfs' : ['wm_prediction', 'wm_tv', 'wm_given']
	},



	'what_to_save_params' : {
		'big_save_keys' : ['um_loss', 'wm_loss', 'wm_prediction', 'wm_tv', 'wm_given'],
		'little_save_keys' : ['um_loss', 'wm_loss'],
		'big_save_len' : 100,
		'big_save_freq' : 10000
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
					'depth' : 2
				},
			'rescale_dict' = {
					'depth' : (64, 64)
				}
		},
		'scene_list' : [environment.example_scene_info],
		'scene_lengths' : [1024 * 32],
		'capacity' : 5
	}

	'train_params' : {
		'updater_func' : train.get_default_updater
	}



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
	raise Exception('FIX TFUTILS TRAINING SAVE')
	train.train_from_params(**train)







