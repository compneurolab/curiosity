'''
Simple 2object_to_1object training main.
'''

import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import numpy as np

from tfutils import base
from curiosity.data.threeworld_data import ThreeWorldDataProvider
import curiosity.models.twoobject_to_oneobject as modelsource

DATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
VALDATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata'
DATA_BATCH_SIZE = 64
MODEL_BATCH_SIZE = 64
TIME_SEEN = 5
SEQUENCE_LEN = 10
CACHE_DIR = '/data/nhaber'
NUM_BATCHES_PER_EPOCH = 115 * 70 * 256 / MODEL_BATCH_SIZE

if not os.path.exists(CACHE_DIR):
	os.mkdir(CACHE_DIR)


def append_it(x, y, step):
	if x is None:
		x = []
	x.append(y)
	return x

def mean_losses_subselect_rest(val_res, skip_num):
	retval = {}
	keys = val_res[0].keys()
	for k in keys:
		if 'loss' in k:
			plucked = [d[k] for d in val_res]
			retval[k] = np.mean(plucked)
		else:
			retval[k] = [val_res[i][k] for i in range(len(val_res)) if i % skip_num == 0]
	return retval


SAVE_TO_GFS = ['normals', 'normals2', 'object_data_seen', 'object_data_future', 'pred', 'actions_seen', 'actions_future']

def grab_all(inputs, outputs, num_to_save = 1, **garbage_params):
	retval = {}	
	batch_size = outputs['normals'].get_shape().as_list()[0]
	for k in SAVE_TO_GFS:
		retval[k] = outputs[k][:num_to_save]
	retval['loss'] = modelsource.l2_loss(outputs)
	return retval






params = {
	'save_params' : {
		'host' : 'localhost',
		'port' : 27017,
		'dbname' : 'future_prediction',
		'collname' : 'choice_2',
		'exp_id' : 'cfg_simple',
		'save_valid_freq' : 2000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 2000,
        'save_initial_filters' : False,
        'cache_dir' : CACHE_DIR,
        'save_to_gfs' : SAVE_TO_GFS
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
			'capacity' : MODEL_BATCH_SIZE * 14 #TODO change!
		},

		'num_steps' : float('inf'),
		'thres_loss' : float('inf')

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
		'learning_rate': .05,
		'decay_rate': 0.95,
		'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
		'staircase': True
	},

	'validation_params' : {
		'valid0' : {
			'data_params' : {
				'func' : ThreeWorldDataProvider,
				'data_path' : VALDATA_PATH,
				'sources' : ['normals', 'normals2', 'actions', 'object_data'],
				'sequence_len' : SEQUENCE_LEN,
				'filters' : ['is_not_teleporting'],
				'shuffle' : True,
				'shuffle_seed' : 0,
				'n_threads' : 1,
				'batch_size' : DATA_BATCH_SIZE			
			},

			'queue_params' : {
				'queue_type' : 'random',
				'batch_size' : MODEL_BATCH_SIZE,
				'seed' : 0,
				'capacity' : MODEL_BATCH_SIZE * 1
			},

			'targets' : {
				'func' : grab_all,
				'targets' : [],
				'num_to_save' : 1,
			},
			'agg_func' : lambda val_res : mean_losses_subselect_rest(val_res, 10),
			'online_agg_func' : append_it,
			'num_steps' : 50
		}





	}

}


if __name__ == '__main__':
	base.get_params()
	base.train_from_params(**params)




