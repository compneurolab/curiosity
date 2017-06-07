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
import curiosity.models.segmentation_models as modelsource


#key params for rapid toggling
EXP_ID_PREFIX = 'simple_seg'
model_func = modelsource.simplest_segmentation
model_cfg_gen = modelsource.gen_cfg_res_preserving_conv
loss_func = modelsource.segmentation_2class_loss

cfg_kwargs = [
	{'num_filters' : [4, 4, 4, 2],
	'sizes' : [3, 3, 3, 3],
	'bypasses' : [None, None, 0, 0]},
	{'num_filters' : [4, 4, 8, 4, 4, 16, 4, 4, 16, 4, 4, 16, 4, 4, 16, 4, 4, 32, 2],
	'sizes' : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
	'bypasses' : [None, None, 0, None, None, [0, -3], None, None, [0, -3], None, None, [0, -3], None, None, [0, -3],
			None, None, [0, -3], [0, -1]]},
	{'num_filters' : [8] * 9 + [2],
	'sizes' : [7] * 10,
	'bypasses' : [None, None, None, None, 0, None, None, None, 0, 0]},
	{'num_filters' : [4] * 5 + [2],
	'sizes' : [7, 7, 5, 5, 3, 3],
	'bypasses' : [None, None, None, None, 0, 0]}	
]

CFGS = [model_cfg_gen(**kwargs) for kwargs in cfg_kwargs]

DESCS = ['dflt', 'resy', 'long', 'desc']

EXP_IDS = [EXP_ID_PREFIX + desc for desc in DESCS]


learning_rate = 1e-5


#key params we likely won't change often
DATA_PATH = '/mnt/fs1/datasets/six_world_dataset/new_tfdata'
VALDATA_PATH = '/mnt/fs1/datasets/six_world_dataset/new_tfvaldata'
STATS_FILE = '/mnt/fs1/datasets/six_world_dataset/new_stats/stats_std.pkl' #should replace this but ok to get started
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
TIME_SEEN = 1
SHORT_LEN = TIME_SEEN
LONG_LEN = TIME_SEEN
MIN_LEN = TIME_SEEN

NUM_BATCHES_PER_EPOCH = 8000 #I have no idea...
IMG_HEIGHT = 128
IMG_WIDTH = 170
COLLNAME = 'segmentation'


N_GPUS = 4
CACHE_DIR_PARENT = 'mnt/fs0/nhaber/cache'




def append_it(x, y, step):
        if x is None:
                x = []
        x.append(y)
        return x

def just_keep_everything(val_res):
        keys = val_res[0].keys()
        return dict((k, [d[k] for d in val_res]) for k in keys)


SAVE_TO_GFS = ['segmentation', 'pred', 'reference_ids', 'master_filter']

def grab_all(inputs, outputs, num_to_save = 1, gpu_id = 0, **garbage_params):
        retval = {}
        batch_size = outputs['pred'].get_shape().as_list()[0]
        for k in SAVE_TO_GFS:
                if k != 'reference_ids':
                        retval[k] = outputs[k][:num_to_save]
                else:
                        retval[k] = outputs[k]
        retval['loss'] = loss_func(outputs, gpu_id = gpu_id)
        return retval


save_params = [
{
	'host' : 'localhost',
	'port' : 27017,
	'dbname' : 'future_prediction',
	'collname' : COLLNAME,
	'save_valid_freq' : 2000,
	'save_filters_freq' : 30000,
	'cache_filters_freq' : 2000,
	'save_initial_filters' : False,
	'save_to_gfs' : SAVE_TO_GFS

}
] * N_GPUS


model_params = [
{
                'func' : model_func,
#               'cfg' : modelsource.cfg_alt_short_jerk,
                'time_seen' : TIME_SEEN, 
                'normalization_method' : {'object_data' : 'screen_normalize', 'actions' : 'minmax'},
                'stats_file' : STATS_FILE,
}
] * N_GPUS



loss_params = [
{
                'targets' : [],
                'agg_func' : tf.reduce_mean,
                'loss_per_case_func' : loss_func,
                'loss_func_kwargs' : {},
                'loss_per_case_func_params' : {}
},
] * N_GPUS

learning_rate_params = [{
                'func': tf.train.exponential_decay,
                'learning_rate': learning_rate,
                'decay_rate': 1.,
                'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                'staircase': True
        }] * N_GPUS


optimizer_params = [{
                'func': modelsource.ParallelClipOptimizer,
                'optimizer_class': tf.train.AdamOptimizer,
                'clip': True,
        # 'momentum': .9
        }] * N_GPUS

train_params = {
                
                'data_params' : {
                        'func' : ShortLongSequenceDataProvider,
                        'data_path' : DATA_PATH,
                        'short_sources' : ['depths', 'objects'],
                        'long_sources' : ['object_data', 'actions', 'reference_ids'],
                        'short_len' : SHORT_LEN,
                        'long_len' : LONG_LEN,
                        'min_len' : MIN_LEN,
                        'filters' : ['is_not_teleporting', 'is_object_in_view'],
                        'shuffle' : True,
                        'shuffle_seed' : 0,
                        'n_threads' : 4,
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
        
 }



validation_params = [{
                'valid0' : {
                        'data_params' : {
                                'func' : ShortLongSequenceDataProvider,
                                'data_path' : VALDATA_PATH,
                                'short_sources' : ['depths', 'objects'],
                                'long_sources' : ['object_data', 'actions', 'reference_ids'],
                                'short_len' : SHORT_LEN,
                                'long_len' : LONG_LEN,
                                'min_len' : MIN_LEN,
                                'filters' : ['is_not_teleporting', 'is_object_in_view'],
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
                }
	}
] * N_GPUS


for i in range(N_GPUS):
	for params in [save_params, model_params, loss_params, learning_rate_params, optimizer_params,
			validation_params]:
		params[i] = copy.deepcopy(params[i])

for i in range(N_GPUS):
	save_params[i]['exp_id'] = EXP_IDS[i]
	save_params[i]['cache_dir'] = os.path.join(CACHE_DIR_PARENT, EXP_IDS[i])
	loss_params[i]['loss_func_kwargs']['gpu_id'] = i
	model_params[i]['gpu_id'] = i
	optimizer_params[i]['gpu_offset'] = i
	validation_params[i]['valid0']['targets']['gpu_id'] = i
	model_params[i]['cfg'] = CFGS[i]
	
params = {
	'save_params' : save_params,
	'model_params' : model_params,
	'load_params' : copy.deepcopy(save_params),
	'train_params' : train_params,
	'loss_params' : loss_params,
	'learning_rate_params' : learning_rate_params,
	'optimizer_params': optimizer_params,
	'validation_params' : validation_params,
	'inter_op_parallelism_threads': 500
}

	







if __name__ == '__main__':
	base.get_params()
	base.train_from_params(**params)
























