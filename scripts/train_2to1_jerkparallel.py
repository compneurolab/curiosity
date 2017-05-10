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
import curiosity.models.jerk_models as modelsource
import copy

#DATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
#VALDATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata'
DATA_PATH = '/data/two_world_dataset/new_tfdata'
VALDATA_PATH = '/data/two_world_dataset/new_tfvaldata'

N_GPUS = 4
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
TIME_SEEN = 3
SHORT_LEN = TIME_SEEN
LONG_LEN = 4
MIN_LEN = 4
CACHE_DIR = '/mnt/fs0/mrowca'
NUM_BATCHES_PER_EPOCH = 115 * 70 * 256 / MODEL_BATCH_SIZE
STATS_FILE = '/mnt/fs0/datasets/two_world_dataset/statistics/stats_again.pkl'
BIN_FILE = '/mnt/fs0/datasets/two_world_dataset/bin_data_file.pkl'
IMG_HEIGHT = 160
IMG_WIDTH = 375
SCALE_DOWN_HEIGHT = 40
SCALE_DOWN_WIDTH = 94
L2_COEF = 200.
EXP_ID = ['otrnn1', 'otrnn2', 'otrnn3', 'otrnn4']

if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

def table_norot_grab_func(path):
    all_filenames = os.listdir(path)
    rng = np.random.RandomState(seed=0)
    rng.shuffle(all_filenames)
    print('got to file grabber!')
    return [os.path.join(path, fn) for fn in all_filenames \
            if '.tfrecords' in fn and 'TABLE' in fn and ':ROT:' not in fn] \
            #and 'FAST_LIFT' in fn]

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

def grab_all(inputs, outputs, num_to_save = 1, gpu_id = 0, **garbage_params):
    retval = {}
    batch_size = outputs['pred'].get_shape().as_list()[0]
    for k in SAVE_TO_GFS:
        if k != 'reference_ids':
            retval[k] = outputs[k][:num_to_save]
        else:
            retval[k] = outputs[k]
    retval['loss'] = modelsource.softmax_cross_entropy_loss_with_bins([], 
            outputs, BIN_FILE, gpu_id=gpu_id)
    return retval

save_params = [{
    'host' : 'localhost',
    'port' : 27017,
    'dbname' : 'future_prediction',
    'collname' : 'jerk',
    'exp_id' : EXP_ID[0],
    'save_valid_freq' : 2000,
    'save_filters_freq': 30000,
    'cache_filters_freq': 2000,
    'save_initial_filters' : False,
    'cache_dir' : CACHE_DIR,
    'save_to_gfs' : SAVE_TO_GFS
}] * N_GPUS

load_params = [{
    'host' : 'localhost',
    'port' : 27017,
    'dbname' : 'future_prediction',
    'collname': 'new_data',
    'exp_id' : EXP_ID[0],
    'do_restore': False,
    'load_query': None
}] * N_GPUS

model_params = [{
    'func' : modelsource.basic_jerk_model,
    'cfg' : modelsource.cfg_class_jerk,
    'time_seen' : TIME_SEEN,
    'normalization_method' : {
        'object_data' : 'screen_normalize', 
        'actions' : 'standard'},
    'stats_file' : STATS_FILE,
    'image_height' : IMG_HEIGHT,
    'image_width' : IMG_WIDTH,
    'scale_down_height' : SCALE_DOWN_HEIGHT,
    'scale_down_width' : SCALE_DOWN_WIDTH,
    'add_depth_gaussian' : True,
    'include_pose' : False,
    'num_classes': 60.,
    'gpu_id' : 0,
}] * N_GPUS

loss_params = [{
    'targets' : [],
    'agg_func' : modelsource.parallel_reduce_mean,
    'loss_per_case_func' : modelsource.softmax_cross_entropy_loss_with_bins,
    'loss_per_case_func_params' : {'_outputs': 'outputs', '_targets_$all': 'inputs'},
    'loss_func_kwargs' : {'bin_data_file': BIN_FILE, 'gpu_id': 0}, #{'l2_coef' : L2_COEF}
}] * N_GPUS

learning_rate_params = [{
    'func': tf.train.exponential_decay,
    'learning_rate': 1e-4,
    'decay_rate': 0.95,
    'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
    'staircase': True
}] * N_GPUS

optimizer_params = [{
    'func': modelsource.ParallelClipOptimizer,
    'optimizer_class': tf.train.AdamOptimizer,
    'clip': True,
    'gpu_offset': 0,
    #'momentum': .9
}] * N_GPUS

validation_params = [{
    'valid0' : {
        'data_params' : {
            'func' : ShortLongSequenceDataProvider,
            'data_path' : VALDATA_PATH,
            'short_sources' : ['normals', 'normals2', 'images'],
            'long_sources' : ['actions', 'object_data', 'reference_ids'],
            'short_len' : SHORT_LEN,
            'long_len' : LONG_LEN,
            'min_len' : MIN_LEN,
            'filters' : ['is_not_teleporting', 'is_object_there', 
                'is_object_in_view', 'is_object_in_view2'],
            'shuffle' : True,
            'shuffle_seed' : 0,
            'n_threads' : 2,
            'batch_size' : DATA_BATCH_SIZE,
            'is_there_subsetting_rule' : 'just_first',
            'is_in_view_subsetting_rule' : 'last_seen_and_first_not',
            },
        'queue_params' : {
            'queue_type' : 'random',
            'batch_size' : MODEL_BATCH_SIZE,
            'seed' : 0,
            'capacity' : MODEL_BATCH_SIZE * 20,
            'min_after_dequeue': MODEL_BATCH_SIZE * 15
            },
        'targets' : {
            'func' : grab_all,
            'targets' : [],
            'num_to_save' : MODEL_BATCH_SIZE,
            'gpu_id': 0,
            },
        # 'agg_func' : lambda val_res : mean_losses_subselect_rest(val_res, 1),
        'agg_func' : just_keep_everything,
        'online_agg_func' : append_it,
        'num_steps' : 10,
    },
}] * N_GPUS

train_params =  {
    'data_params' : {
        'func' : ShortLongSequenceDataProvider,
        'data_path' : DATA_PATH,
        'short_sources' : ['normals', 'normals2', 'images'],
        'long_sources' : ['actions', 'object_data', 'reference_ids'],
        'short_len' : SHORT_LEN,
        'long_len' : LONG_LEN,
        'min_len' : MIN_LEN,
        'filters' : ['is_not_teleporting', 'is_object_there', 
            'is_object_in_view', 'is_object_in_view2'],
        'shuffle' : True,
        'shuffle_seed' : 0,
        'n_threads' : 4,
        'batch_size' : DATA_BATCH_SIZE,
        'file_grab_func' : table_norot_grab_func,
        'is_there_subsetting_rule' : 'just_first',
        'is_in_view_subsetting_rule' : 'last_seen_and_first_not',
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

for i, _ in enumerate(save_params):
    save_params[i] = copy.deepcopy(save_params[i])
    load_params[i] = copy.deepcopy(load_params[i])
    model_params[i] = copy.deepcopy(model_params[i])
    loss_params[i] = copy.deepcopy(loss_params[i])
    learning_rate_params[i] = copy.deepcopy(learning_rate_params[i])
    optimizer_params[i] = copy.deepcopy(optimizer_params[i])
    validation_params[i] = copy.deepcopy(validation_params[i])

for i, _ in enumerate(model_params):
    save_params[i]['exp_id'] = EXP_ID[i]
    load_params[i]['exp_id'] = EXP_ID[i]
    
    loss_params[i]['loss_func_kwargs']['gpu_id'] = i
    model_params[i]['gpu_id'] = i
    optimizer_params[i]['gpu_offset'] = i
    validation_params[i]['valid0']['targets']['gpu_id'] = i

params = {
    'save_params' : save_params,
    'load_params' : load_params,
    'model_params' : model_params,
    'train_params' : train_params,
    'loss_params' : loss_params,
    'learning_rate_params' : learning_rate_params,
    'optimizer_params': optimizer_params,
    'validation_params' : validation_params,
    'inter_op_parallelism_threads': 500,
}

if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
