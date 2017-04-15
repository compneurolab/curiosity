'''
Simple 2object_to_1object training main.
'''

import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')

from tfutils import base
from curiosity.data.threeworld_data import ThreeWorldDataProvider
import curiosity.models.twoobject_to_oneobject as modelsource

DATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
VALDATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata'
DATA_BATCH_SIZE = 32
MODEL_BATCH_SIZE = 32
TIME_SEEN = 5
SEQUENCE_LEN = 10
CACHE_DIR = '/data/nhaber'
NUM_BATCHES_PER_EPOCH = 115 * 70 * 256 / MODEL_BATCH_SIZE

if not os.path.exists(CACHE_DIR):
	os.mkdir(CACHE_DIR)

params = {
	'save_params' : {
		'host' : 'localhost',
		'port' : 27017,
		'dbname' : 'future_prediction',
		'collname' : 'choice_2',
		'exp_id' : 'test',
		'save_valid_freq' : 2000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 2000,
        'save_initial_filters' : False,
        'cache_dir' : CACHE_DIR
	},

	'model_params' : {
		'func' : modelsource.simple_conv_to_mlp_structure,
		'cfg' : modelsource.cfg_simple,
		'time_seen' : TIME_SEEN,
	},

	'train_params' : {

		'data_params' : {
			'func' : ThreeWorldDataProvider,
			'data_path' : DATA_PATH,
			'sources' : ['normals', 'normals2', 'actions', 'object_data'],
			'sequence_len' : SEQUENCE_LEN,
			'filters' : ['is_not_teleporting'],
			'shuffle' : True,
			'shuffle_seed' : 0,
			'n_threads' : 4,
			'batch_size' : DATA_BATCH_SIZE
		},

		'queue_params' : {
			'queue_type' : 'random',
			'batch_size' : MODEL_BATCH_SIZE,
			'seed' : 0,
			'capacity' : MODEL_BATCH_SIZE * 2 #TODO change!
		}

	},

	'loss_params' : {
		'targets' : [],
		'agg_func' : tf.reduce_mean,
		'loss_per_case_func' : modelsource.l2_loss,
		'loss_func_kwargs' : {},
		'loss_per_case_func_params' : {}
	},

	'learning_rate_params': {
		'func': tf.train.exponential_decay,
		'learning_rate': 1e-3,
		'decay_rate': 0.95,
		'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
		'staircase': True
	},

	'validation_params' : {}

}


if __name__ == '__main__':
	base.get_params()
	base.train_from_params(**params)




