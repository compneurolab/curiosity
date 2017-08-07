'''
Now with new data provider, and 2->2 architecture.
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

CACHE_NUM = 2
LOCAL = False
if LOCAL:
    DATA_PATH = '/data2/mrowca/datasets/eight_world_dataset/tfdata'
    VALDATA_PATH = '/data2/mrowca/datasets/eight_world_dataset/tfvaldata'
    CACHE_DIR = '/data2/mrowca/cache' + str(CACHE_NUM)
    STATS_FILE = '/data2/mrowca/datasets/eight_world_dataset/new_stats/stats_std.pkl'
else:
    DATA_PATH = '/mnt/fs1/datasets/eight_world_dataset/tfdata'
    VALDATA_PATH = '/mnt/fs1/datasets/eight_world_dataset/tfvaldata'
    CACHE_DIR = '/mnt/fs0/mrowca/cache' + str(CACHE_NUM)
    STATS_FILE = '/mnt/fs1/datasets/eight_world_dataset/new_stats/stats_std.pkl'
BIN_PATH = '' #'/mnt/fs1/datasets/eight_world_dataset/'
BIN_FILE = '' #'/mnt/fs1/datasets/eight_world_dataset/bin_data_file.pkl'

N_GPUS = 1
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 64
TIME_SEEN = 1 #2
SHORT_LEN = TIME_SEEN
LONG_LEN = 1 #3
MIN_LEN = 1 #3
NUM_BATCHES_PER_EPOCH = 4000 * 256 / MODEL_BATCH_SIZE
IMG_HEIGHT = 128
IMG_WIDTH = 170
SCALE_DOWN_HEIGHT = 64
SCALE_DOWN_WIDTH = 88
L2_COEF = 200.
EXP_ID = ['flex2dBott', 
'flexBott',
'flex2d', 
'flex']
#EXP_ID = ['res_jerk_eps', 'map_jerk_eps', 'sym_jerk_eps', 'bypass_jerk_eps']
LRS = [0.001, 0.001, 0.001, 0.001]
n_classes = 3
buckets = 0
min_particle_distance = 0.01
DEPTH_DIM = 32
CFG = [
        #modelsource.particle_2d_bottleneck_cfg(n_classes * DEPTH_DIM, nonlin='relu'),
        #modelsource.particle_bottleneck_cfg(n_classes, nonlin='relu'),
        modelsource.particle_2d_cfg(n_classes * DEPTH_DIM, nonlin='relu'),
        #modelsource.particle_cfg(n_classes, nonlin='relu'),
        ]
CACHE_DIRS = [CACHE_DIR + str(d) for d in range(4)]
SEED = 4

if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

def table_norot_grab_func(path):
    all_filenames = os.listdir(path)
    rng = np.random.RandomState(seed=SEED)
    rng.shuffle(all_filenames)
    print('got to file grabber!')
    return [os.path.join(path, fn) for fn in all_filenames \
            if fn.endswith('.tfrecords')] 
            #and fn.startswith('2:')] \
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


SAVE_TO_GFS = []#['pred_vel_flat', 'reference_ids']

def grab_all(inputs, outputs, bin_file = BIN_FILE, 
        num_to_save = 1, gpu_id = 0, **garbage_params):
    retval = {}
    batch_size = outputs['prediction'].get_shape().as_list()[0]
    retval['loss'] = modelsource.flex_loss( 
            outputs, gpu_id=gpu_id, min_particle_distance=min_particle_distance)
    for k in SAVE_TO_GFS:
        if k != 'reference_ids':
            if k in ['pred_vel_1', 'pred_next_vel_1', 'pred_next_img_1',
                    'pred_delta_vel_1']:
                pred = outputs[k]
		shape = pred.get_shape().as_list()
		pred = tf.reshape(pred, shape[0:3] + [3, shape[3] / 3])
                pred = tf.cast(tf.argmax(
                    pred, axis=tf.rank(pred) - 1), tf.uint8)[:num_to_save]
		#pred = sample_from_discretized_mix_logistic(pred, n_classes/10, 
                #        buckets=buckets)
		retval[k] = pred
            elif k == 'depths_raw':
                depths = outputs[k][:num_to_save]
                retval[k] = depths[:,-1,:,:,0]
            elif k in ['jerks']:
                jerks = outputs[k][:num_to_save]
                retval[k] = jerks[:,-1]
            elif k in ['vels']:
                vels = outputs[k][:num_to_save]
                retval[k] = vels[:,2]
            else:
                retval[k] = outputs[k][:num_to_save]
        else:
            retval[k] = outputs[k]
    #filter examples
    filter_examples = False
    if filter_examples:
        thres = 0.5412
        mask = tf.norm(outputs['jerk_all'][:num_to_save], ord='euclidean', axis=2)
        mask = tf.logical_or(tf.greater(mask[:,0], thres),
                tf.greater(mask[:,1], thres))
        for k in SAVE_TO_GFS:
            retval[k] = tf.gather(retval[k], tf.where(mask))

    return retval

save_params = [{
    'host' : 'localhost',
    'port' : 24444,
    'dbname' : 'future_prediction',
    'collname' : 'flex',
    'exp_id' : EXP_ID[0],
    'save_valid_freq' : 4000,
    'save_filters_freq': 30000,
    'cache_filters_freq': 4000,
    'save_metrics_freq': 1000,
    'save_initial_filters' : False,
    'cache_dir' : CACHE_DIR,
    'save_to_gfs' : SAVE_TO_GFS
}] * N_GPUS

load_params = [{
    'host' : 'localhost',
    'port' : 24444,
    'dbname' : 'future_prediction',
    'collname': 'flex',
    'exp_id' : EXP_ID[0],
    'do_restore': False,
    'load_query': None
}] * N_GPUS

model_params = [{
    'func' : modelsource.flex_model,
    'cfg' : CFG[0],
    'time_seen' : TIME_SEEN,
    'normalization_method' : {
        'actions' : 'minmax'},
    'stats_file' : STATS_FILE,
    'image_height' : IMG_HEIGHT,
    'image_width' : IMG_WIDTH,
    #'num_classes': 60.,
    'gpu_id' : 0,
}] * N_GPUS

loss_params = [{
    'targets' : [],
    'agg_func' : modelsource.parallel_reduce_mean,
    'loss_per_case_func' : modelsource.flex_loss,
    'loss_per_case_func_params' : {'_outputs': 'outputs', '_targets_$all': 'inputs'},
    'loss_func_kwargs' : {'gpu_id': 0, 'min_particle_distance': min_particle_distance}, 
    #{'l2_coef' : L2_COEF}
}] * N_GPUS

learning_rate_params = [{
    'func': tf.train.exponential_decay,
    'learning_rate': LRS[0],
    'decay_rate': 0.95,
    'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
    'staircase': True
}] * N_GPUS

optimizer_params = [{
    'func': modelsource.ParallelClipOptimizer,
    'optimizer_class': tf.train.RMSPropOptimizer, #tf.train.AdamOptimizer,
    'clip': True,
    'gpu_offset': 0,
    #'momentum': .9,
    #'beta1': 0.95,
    #'beta2': 0.9995,
}] * N_GPUS

validation_params = [{
    'valid0' : {
        'data_params' : {
            'func' : ShortLongSequenceDataProvider,
            'data_path' : VALDATA_PATH,
            'short_sources' : [], #'depths2', 'normals2', 'images'
            'long_sources' : ['actions', #'depths', 'objects', 
                    'object_data', 'reference_ids', 'max_coordinates', 'min_coordinates', \
                    'grid_32'],
            'short_len' : SHORT_LEN,
            'long_len' : LONG_LEN,
            'min_len' : MIN_LEN,
            'filters' : ['is_moving', 'is_object_in_view'],
            'shuffle' : True,
            'shuffle_seed' : SEED,
            'n_threads' : 2,
            'batch_size' : DATA_BATCH_SIZE,
            'file_grab_func' : table_norot_grab_func,
           # 'is_there_subsetting_rule' : 'just_first',
            'is_in_view_subsetting_rule' : 'first_there',
            },
        'queue_params' : {
            'queue_type' : 'fifo',
            'batch_size' : MODEL_BATCH_SIZE,
            'seed' : SEED,
            'capacity' : MODEL_BATCH_SIZE * 30,
            'min_after_dequeue': MODEL_BATCH_SIZE * 10
            },
        'targets' : {
            'func' : grab_all,
            'targets' : [],
            'num_to_save' : 10, #MODEL_BATCH_SIZE,
            'gpu_id': 0,
            'bin_file': BIN_FILE,
            },
        # 'agg_func' : lambda val_res : mean_losses_subselect_rest(val_res, 1),
        'agg_func' : just_keep_everything,
        'online_agg_func' : append_it,
        'num_steps' : round(256.0 * 12 * 4 / MODEL_BATCH_SIZE),
    },
}] * N_GPUS

train_params =  {
    'validate_first': False,
    'data_params' : {
        'func' : ShortLongSequenceDataProvider,
        'data_path' : DATA_PATH,
        'short_sources' : [], #'depths2', 'normals2', 'images' 
        'long_sources' : ['actions', #'depths', 'objects', 
                'object_data', 'reference_ids', 'max_coordinates', 'min_coordinates', \
                'grid_32'],
        'short_len' : SHORT_LEN,
        'long_len' : LONG_LEN,
        'min_len' : MIN_LEN,
        'filters' : ['is_moving', 'is_object_in_view'],
        'shuffle' : True,
        'shuffle_seed' : SEED,
        'n_threads' : 4,
        'batch_size' : DATA_BATCH_SIZE,
        'file_grab_func' : table_norot_grab_func,
        #'is_there_subsetting_rule' : 'just_first',
        'is_in_view_subsetting_rule' : 'first_there',
    },
        
    'queue_params' : {
        'queue_type' : 'random',
        'batch_size' : MODEL_BATCH_SIZE,
        'seed' : SEED,
        'capacity' : MODEL_BATCH_SIZE * 60
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

    save_params[i]['cache_dir'] = CACHE_DIRS[i]
    loss_params[i]['loss_func_kwargs']['gpu_id'] = i
    #loss_params[i]['loss_func_kwargs']['bin_data_file'] = BIN_PATH + EXP_ID[i] + '.pkl'
    model_params[i]['gpu_id'] = i
    optimizer_params[i]['gpu_offset'] = i
    validation_params[i]['valid0']['targets']['gpu_id'] = i
    #validation_params[i]['valid0']['targets']['bin_file'] = BIN_PATH + EXP_ID[i] + '.pkl'
    model_params[i]['cfg'] = CFG[i]
    learning_rate_params[i]['learning_rate'] = LRS[i]

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
