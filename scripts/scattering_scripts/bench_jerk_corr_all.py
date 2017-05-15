'''
Correlation loss 1 to 1 benchmark for scatter dataset
'''


import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import numpy as np
import copy


from tfutils import base, optimizer
from curiosity.data.short_long_sequence_data import ShortLongSequenceDataProvider
import curiosity.models.jerk_models as modelsource


#key params for rapid toggling
EXP_ID_PREFIX = 'bench_jc_norm'
model_func = modelsource.basic_jerk_bench
model_cfg_gen = modelsource.gen_cfg_jerk_bench
learning_rate = 1e-5
widths = [1000, 2000, 3000]
drop_keep = .75
depths = [2, 3, 4, 5]


#key params we likely won't change often
DATA_PATH = '/mnt/fs0/datasets/three_world_dataset/new_tfdata_newobj'
VALDATA_PATH = '/mnt/fs0/datasets/three_world_dataset/new_tfvaldata_newobj'
STATS_FILE = '/mnt/fs0/datasets/two_world_dataset/statistics/stats_again.pkl' #should replace this but ok to get started
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
TIME_SEEN = 3
SHORT_LEN = TIME_SEEN
LONG_LEN = 4
MIN_LEN = 4

NUM_BATCHES_PER_EPOCH = 2 * 1000 #I think...
IMG_HEIGHT = 128
IMG_WIDTH = 170
SCALE_DOWN_HEIGHT = 32
SCALE_DOWN_WIDTH = 43
L2_COEF = 200.
COLLNAME = 'scatter'


def append_it(x, y, step):
	if x is None:
		x = []
	x.append(y)
	return x

def just_keep_everything(val_res):
	keys = val_res[0].keys()
	return dict((k, [d[k] for d in val_res]) for k in keys)

SAVE_TO_GFS = ['object_data_future', 'pred', 'object_data_seen_1d', 'reference_ids', 'master_filter']

def grab_all(inputs, outputs, num_to_save = 1, **garbage_params):
	retval = {}
	batch_size = outputs['pred'].get_shape().as_list()[0]
	for k in SAVE_TO_GFS:
		if k != 'reference_ids':
			retval[k] = outputs[k][:num_to_save]
		else:
			retval[k] = outputs[k]
	retval['loss'] = modelsource.correlation_jerk_loss(outputs, l2_coef = L2_COEF)
	return retval


params = {
	'save_params' : {
		'host' : 'localhost',
		'port' : 27017,
		'dbname' : 'future_prediction',
		'collname' : COLLNAME,
#		'exp_id' : EXP_ID,
		'save_valid_freq' : 2000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 2000,
        'save_initial_filters' : False,
#        'cache_dir' : CACHE_DIR,
        'save_to_gfs' : SAVE_TO_GFS
	},

	'model_params' : {
		'func' : model_func,
#		'cfg' : model_cfg,
		'time_seen' : TIME_SEEN,
		'normalization_method' : {'object_data' : 'screen_normalize', 'actions' : 'minmax'},
		'image_height' : IMG_HEIGHT,
		'image_width' : IMG_WIDTH,
		'scale_down_height' : SCALE_DOWN_HEIGHT,
		'scale_down_width' : SCALE_DOWN_WIDTH,
		'stats_file' : STATS_FILE
	},

	'train_params' : {

		'data_params' : {
			'func' : ShortLongSequenceDataProvider,
			'data_path' : DATA_PATH,
			'short_sources' : [],
			'long_sources' : ['actions', 'object_data', 'reference_ids'],
			'short_len' : SHORT_LEN,
			'long_len' : LONG_LEN,
			'min_len' : MIN_LEN,
			'filters' : ['is_not_teleporting'],
			'shuffle' : True,
			'shuffle_seed' : 0,
			'n_threads' : 4,
			'batch_size' : DATA_BATCH_SIZE,
		},

		'queue_params' : {
			'queue_type' : 'random',
			'batch_size' : MODEL_BATCH_SIZE,
			'seed' : 0,
			'capacity' : MODEL_BATCH_SIZE * 40 #TODO change!
		},

		'num_steps' : 200000,
		'thres_loss' : float('inf')

	},

	'loss_params' : {
		'targets' : [],
		'agg_func' : tf.reduce_mean,
		'loss_per_case_func' : modelsource.correlation_jerk_loss,
		'loss_func_kwargs' : {'l2_coef' : L2_COEF},
		'loss_per_case_func_params' : {}
	},

	'learning_rate_params': {
		'func': tf.train.exponential_decay,
		'learning_rate': learning_rate,
		'decay_rate': 1.,
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
				'short_sources' : [],
				'long_sources' : ['actions', 'object_data', 'reference_ids'],
				'short_len' : SHORT_LEN,
				'long_len' : LONG_LEN,
				'min_len' : MIN_LEN,
				'filters' : ['is_not_teleporting'],
				'shuffle' : True,
				'shuffle_seed' : 0,
				'n_threads' : 1,
				'batch_size' : DATA_BATCH_SIZE,
			},

			'queue_params' : {
				'queue_type' : 'random',
				'batch_size' : MODEL_BATCH_SIZE,
				'seed' : 0,
				'capacity' : 20 * MODEL_BATCH_SIZE
			},

			'targets' : {
				'func' : grab_all,
				'targets' : [],
				'num_to_save' : MODEL_BATCH_SIZE,
			},
			# 'agg_func' : lambda val_res : mean_losses_subselect_rest(val_res, 1),
			'agg_func' : just_keep_everything,
			'online_agg_func' : append_it,
			'num_steps' : 50
		},

	}

}


if __name__ == '__main__':
	base.get_params()
	for w in widths:
		for d in depths:
			params_copy = copy.deepcopy(params)
			EXP_ID = EXP_ID_PREFIX + '_' + str(w) + '_' + str(d)
			CACHE_DIR = os.path.join('/mnt/fs0/nhaber/cache', EXP_ID)	
			if not os.path.exists(CACHE_DIR):
				os.mkdir(CACHE_DIR)
			cfg = model_cfg_gen(depth = d, width = w, drop_keep = drop_keep)
			print(cfg)
			params_copy['model_params']['cfg'] = cfg
			params_copy['save_params']['exp_id'] = EXP_ID
			params_copy['save_params']['cache_dir'] = CACHE_DIR
			base.train_from_params(**params_copy)




