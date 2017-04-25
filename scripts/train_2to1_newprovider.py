'''
Now with new data provider, and 2->1 architecture.
'''


import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import numpy as np

from tfutils import base, optimizer
from curiosity.data.short_long_sequence_data import ShortLongSequenceDataProvider
import curiosity.models.twoobject_to_oneobject as modelsource

DATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
VALDATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata'
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
TIME_SEEN = 3
SHORT_LEN = TIME_SEEN
LONG_LEN = 23
MIN_LEN = 6
CACHE_DIR = '/data/nhaber'
NUM_BATCHES_PER_EPOCH = 115 * 70 * 256 / MODEL_BATCH_SIZE
STATS_FILE = '/mnt/fs0/datasets/two_world_dataset/statistics/stats_updated.pkl'
IMG_HEIGHT = 160
IMG_WIDTH = 375
SCALE_DOWN_HEIGHT = 40
SCALE_DOWN_WIDTH = 94

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
		elif 'reference_ids' in k:
			retval[k] = [d[k] for d in val_res]
		else:
			retval[k] = [val_res[i][k] for i in range(len(val_res)) if i % skip_num == 0]
	return retval


SAVE_TO_GFS = ['object_data_future', 'pred', 'object_data_seen_1d', 'reference_ids', 'master_filter']

def grab_all(inputs, outputs, num_to_save = 1, **garbage_params):
	retval = {}
	batch_size = outputs['pred'].get_shape().as_list()[0]
	for k in SAVE_TO_GFS:
		if k != 'reference_ids':
			retval[k] = outputs[k][:num_to_save]
		else:
			retval[k] = outputs[k]
	retval['loss'] = modelsource.diff_loss_with_mask(outputs)
	return retval



#cfg_simple lr .05, no normalization
#cfg_simple_norm lr .05, normalization
#cfg_2: lr .05, normalization, diff loss, try some rmsprop
#cfg_2_lr-3, lr .001
#cfg_2_rmsprop lr .001 now with rmsprop
#rms_-4 3123
#big_lr rms, lr .05
#fixed_end fixed end nonlinearity, otherwise like big_lr
#rms_-4_fixed 
#rms_5-2_fixed rms_prop
#rms_1-5_fixed lr 1-05
#rms_1-6_fixed
#nrmfx_5-2

params = {
	'save_params' : {
		'host' : 'localhost',
		'port' : 27017,
		'dbname' : 'future_prediction',
		'collname' : 'choice_2',
		'exp_id' : 'res18_adam',
		'save_valid_freq' : 2000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 2000,
        'save_initial_filters' : False,
        'cache_dir' : CACHE_DIR,
        'save_to_gfs' : SAVE_TO_GFS
	},

	'model_params' : {
		'func' : modelsource.one_to_two_to_one,
		'cfg' : modelsource.cfg_one_to_two_to_one,
		'time_seen' : TIME_SEEN,
		'normalization_method' : {'object_data' : 'screen_normalize', 'actions' : 'standard'},
		'stats_file' : STATS_FILE,
		'image_height' : IMG_HEIGHT,
		'image_width' : IMG_WIDTH,
		'scale_down_height' : SCALE_DOWN_HEIGHT,
		'scale_down_width' : SCALE_DOWN_WIDTH
	},

	'train_params' : {

		'data_params' : {
			'func' : ShortLongSequenceDataProvider,
			'data_path' : DATA_PATH,
			'short_sources' : ['normals', 'normals2'],
			'long_sources' : ['actions', 'object_data', 'reference_ids'],
			'short_len' : SHORT_LEN,
			'long_len' : LONG_LEN,
			'min_len' : MIN_LEN,
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
			'capacity' : MODEL_BATCH_SIZE * 40 #TODO change!
		},

		'num_steps' : float('inf'),
		'thres_loss' : float('inf')

	},

	'loss_params' : {
		'targets' : [],
		'agg_func' : tf.reduce_mean,
		'loss_per_case_func' : modelsource.diff_loss_with_mask,
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

	'optimizer_params': {
		'func': optimizer.ClipOptimizer,
		'optimizer_class': tf.train.AdamOptimizer,
		'clip': True,
	# 'momentum': .9
	},


	'validation_params' : {
		'valid0' : {
			'data_params' : {
				'func' : ShortLongSequenceDataProvider,
				'data_path' : VALDATA_PATH,
				'short_sources' : ['normals', 'normals2'],
				'long_sources' : ['actions', 'object_data', 'reference_ids'],
				'short_len' : SHORT_LEN,
				'long_len' : LONG_LEN,
				'min_len' : MIN_LEN,
				'filters' : ['is_not_teleporting'],
				'shuffle' : True,
				'shuffle_seed' : 0,
				'n_threads' : 2,
				'batch_size' : DATA_BATCH_SIZE
			},

			'queue_params' : {
				'queue_type' : 'random',
				'batch_size' : MODEL_BATCH_SIZE,
				'seed' : 0,
				'capacity' : MODEL_BATCH_SIZE * 20
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




