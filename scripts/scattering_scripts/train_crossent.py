'''
Cross entropy, tried to make bins roughly equivalent, 2 to 1 loss.
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
import curiosity.models.jerk_models as modelsource


#key params for rapid toggling
EXP_ID_PREFIX = 'disc_eq_bins'
model_func = modelsource.basic_jerk_model
model_cfg_gen = modelsource.gen_cfg_short_jerk

learning_rate = 1e-5
#filters at end, encode depth, features in hidden, hidden depth
some_cfg_tuples = [(24, 34, 34, 2, 250, 3), (16, 16, 32, 2, 2000, 3), (16, 16, 16, 2, 3000, 3), (16, 16, 32, 2, 3000, 4), (16, 32, 32, 3, 3000, 4)]
drop_keep = .75


#key params we likely won't change often
DATA_PATH = '/mnt/fs0/datasets/three_world_dataset/new_tfdata_newobj'
VALDATA_PATH = '/mnt/fs0/datasets/three_world_dataset/new_tfvaldata_newobj'
STATS_FILE = '/mnt/fs0/datasets/three_world_dataset/stats_std.pkl' #should replace this but ok to get started
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
TIME_SEEN = 3
SHORT_LEN = TIME_SEEN
LONG_LEN = 4
MIN_LEN = 4

NUM_BATCHES_PER_EPOCH = 4 * 1000 #I think...
IMG_HEIGHT = 128
IMG_WIDTH = 170
SCALE_DOWN_HEIGHT = 32
SCALE_DOWN_WIDTH = 43
NUM_CLASSES = 22
BINNING_FILE = os.path.join('/mnt/fs0/nhaber/cross_ent_bins', 'more_balanced_classes_try.pkl')
COLLNAME = 'scatter'

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
	retval['loss'] = modelsource.softmax_cross_entropy_loss_with_bins(outputs, bin_data_file = BINNING_FILE)
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
		'collname' : COLLNAME,
#		'exp_id' : 'jerk_corr',
		'save_valid_freq' : 2000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 2000,
        'save_initial_filters' : False,
#        'cache_dir' : CACHE_DIR,
        'save_to_gfs' : SAVE_TO_GFS
	},

	'model_params' : {
		'func' : model_func,
#		'cfg' : modelsource.cfg_alt_short_jerk,
		'time_seen' : TIME_SEEN,
		'normalization_method' : {'object_data' : 'screen_normalize', 'actions' : 'standard'},
		'stats_file' : STATS_FILE,
		'image_height' : IMG_HEIGHT,
		'image_width' : IMG_WIDTH,
		'scale_down_height' : SCALE_DOWN_HEIGHT,
		'scale_down_width' : SCALE_DOWN_WIDTH,
		'add_depth_gaussian' : True,
		'include_pose' : False,
		'depths_not_normals_images' : True,
		'depth_cutoff' : 17.32,
		'num_classes' : NUM_CLASSES
	},

	'train_params' : {

		'data_params' : {
			'func' : ShortLongSequenceDataProvider,
			'data_path' : DATA_PATH,
			'short_sources' : ['depths', 'depths2'],
			'long_sources' : ['actions', 'object_data', 'reference_ids'],
			'short_len' : SHORT_LEN,
			'long_len' : LONG_LEN,
			'min_len' : MIN_LEN,
			'filters' : ['is_not_teleporting', 'is_object_in_view', 'is_object_in_view2'],
			'shuffle' : True,
			'shuffle_seed' : 0,
			'n_threads' : 1,
			'batch_size' : DATA_BATCH_SIZE,
			'is_in_view_subsetting_rule' : 'both_there'
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
		'loss_per_case_func' : modelsource.softmax_cross_entropy_loss_with_bins,
		'loss_func_kwargs' : {'bin_data_file' : BINNING_FILE},
		'loss_per_case_func_params' : {}
	},
	
	'learning_rate_params': {
		'func': tf.train.exponential_decay,
		'learning_rate': 1e-5,
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
				'short_sources' : ['depths', 'depths2'],
				'long_sources' : ['actions', 'object_data', 'reference_ids'],
				'short_len' : SHORT_LEN,
				'long_len' : LONG_LEN,
				'min_len' : MIN_LEN,
				'filters' : ['is_not_teleporting', 'is_object_in_view', 'is_object_in_view2'],
				'shuffle' : True,
				'shuffle_seed' : 0,
				'n_threads' : 1,
				'batch_size' : DATA_BATCH_SIZE,
				'is_in_view_subsetting_rule' : 'both_there'
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

#		'valid1' : {
#			'data_params' : {
#				'func' : ShortLongSequenceDataProvider,
#				'data_path' : DATA_PATH,
#				'short_sources' : ['normals', 'normals2', 'images', 'images2', 'objects', 'objects2'],
#				'long_sources' : ['actions', 'object_data', 'reference_ids'],
#				'short_len' : SHORT_LEN,
#				'long_len' : LONG_LEN,
#				'min_len' : MIN_LEN,
#				'filters' : ['is_not_teleporting', 'is_object_there'],
#				'shuffle' : True,
#				'shuffle_seed' : 0,
#				'n_threads' : 1,
#				'batch_size' : DATA_BATCH_SIZE,
#				'file_grab_func' : table_norot_grab_func,
#				'is_there_subsetting_rule' : 'just_first'
#			},
#
#			'queue_params' : {
#				'queue_type' : 'fifo',
#				'batch_size' : MODEL_BATCH_SIZE,
#				'seed' : 0,
#				'capacity' : MODEL_BATCH_SIZE
#			},
#
#			'targets' : {
#				'func' : grab_all,
#				'targets' : [],
#				'num_to_save' : MODEL_BATCH_SIZE,
#			},
#			# 'agg_func' : lambda val_res : mean_losses_subselect_rest(val_res, 1),
#			'agg_func' : just_keep_everything,
#			'online_agg_func' : append_it,
#			'num_steps' : 20
#		}






	}

}

#filters at end, encode depth, features in hidden, hidden depth

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
	cfg_version = int(sys.argv[2])
	first_filters, encode_filters, end_filters, encode_depth, hidden_num_features, hidden_depth = some_cfg_tuples[cfg_version]
	cfg = model_cfg_gen(num_filters_before_concat = first_filters, num_filters_after_concat = encode_filters, num_filters_together = end_filters, encode_depth = encode_depth, hidden_num_features = hidden_num_features, hidden_depth = hidden_depth, num_classes = NUM_CLASSES)
	params['model_params']['cfg'] = cfg
	EXP_ID = EXP_ID_PREFIX + '_' + str(cfg_version)
	params['save_params']['exp_id'] = EXP_ID
	CACHE_DIR = os.path.join('/mnt/fs0/nhaber', EXP_ID)
	params['save_params']['cache_dir'] = CACHE_DIR
	base.train_from_params(**params)




