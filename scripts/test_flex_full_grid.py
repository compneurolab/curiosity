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
from tqdm import trange

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
MODEL_BATCH_SIZE = 256 #64
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
EXP_ID = [#'flex2dBott_4', 
'flexBott_4',
#'flex2d_4', 
#'flex_4',
]
#EXP_ID = ['res_jerk_eps', 'map_jerk_eps', 'sym_jerk_eps', 'bypass_jerk_eps']
LRS = [0.001, 0.001, 0.001, 0.001]
n_classes = 3
buckets = 0
min_particle_distance = 0.01
DEPTH_DIM = 32
CFG = [
        #modelsource.particle_2d_bottleneck_cfg(n_classes * DEPTH_DIM, nonlin='relu'),
        modelsource.particle_bottleneck_cfg(n_classes, nonlin='relu'),
        #modelsource.particle_2d_cfg(n_classes * DEPTH_DIM, nonlin='relu'),
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

def keep_all(step_results):
    return step_results[0]

SAVE_TO_GFS = []#['pred_vel_flat', 'reference_ids']

def grab_all(inputs, outputs, bin_file = BIN_FILE, 
        num_to_save = 1, gpu_id = 0, **garbage_params):
    retval = {
            'prediction': outputs['prediction'],
            'next_velocity': outputs['next_velocity'],
            'full_grids': outputs['full_grids'],
            'grid_placeholder': outputs['grid_placeholder'],
            'actions': outputs['actions'],
            'in_view': outputs['in_view'],
            'is_moving': outputs['is_moving'],
            }
    return retval

save_params = [{
    'host' : 'localhost',
    'port' : 24444,
    'dbname' : 'future_prediction',
    'collname' : 'flex',
    'exp_id' : EXP_ID[0],
    'save_valid_freq' : np.round(256 * 84 * 4 / MODEL_BATCH_SIZE).astype(np.int32),
    'save_filters_freq': np.round(256 * 84 * 4 / MODEL_BATCH_SIZE * 10).astype(np.int32),
    'cache_filters_freq': np.round(256 * 84 * 4 / MODEL_BATCH_SIZE).astype(np.int32),
    'save_metrics_freq': np.round(256 * 84 * 4 / MODEL_BATCH_SIZE / 10).astype(np.int32),
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
    'do_restore': True,
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
    'my_test' : True,
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
                    'grid_32', 'is_moving', 'is_object_in_view'],
            'short_len' : SHORT_LEN,
            'long_len' : LONG_LEN,
            'min_len' : MIN_LEN,
            'filters' : None, #['is_moving', 'is_object_in_view'],
            'shuffle' : True,
            'shuffle_seed' : SEED,
            'n_threads' : 1,
            'batch_size' : DATA_BATCH_SIZE,
            'file_grab_func' : table_norot_grab_func,
           # 'is_there_subsetting_rule' : 'just_first',
            #'is_in_view_subsetting_rule' : 'first_there',
            },
        'queue_params' : {
            'queue_type' : 'fifo',
            'batch_size' : MODEL_BATCH_SIZE,
            'seed' : SEED,
            'capacity' : MODEL_BATCH_SIZE * 11,
            'min_after_dequeue': MODEL_BATCH_SIZE * 10
            },
        'targets' : {
            'func' : grab_all,
            'targets' : [],
            'num_to_save' : 5,
            'gpu_id': 0,
            'bin_file': BIN_FILE,
            },
        # 'agg_func' : lambda val_res : mean_losses_subselect_rest(val_res, 1),
        'agg_func' : keep_all,
        #'online_agg_func' : append_it,
        'num_steps' : 1, #np.round(256.0 * 12 * 4 / MODEL_BATCH_SIZE).astype(np.int32),
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
        'filters' : None, #['is_moving', 'is_object_in_view'],
        'shuffle' : True,
        'shuffle_seed' : SEED,
        'n_threads' : 1,
        'batch_size' : DATA_BATCH_SIZE,
        'file_grab_func' : table_norot_grab_func,
        #'is_there_subsetting_rule' : 'just_first',
        #'is_in_view_subsetting_rule' : 'first_there',
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
    'save_params' : save_params[0],
    'load_params' : load_params[0],
    'model_params' : model_params[0],
    #'train_params' : train_params,
    #'loss_params' : loss_params,
    #'learning_rate_params' : learning_rate_params,
    #'optimizer_params': optimizer_params,
    'validation_params' : validation_params[0],
    'inter_op_parallelism_threads': 500,
    'dont_run': True,
}

if __name__ == '__main__':
    # get session and data
    print('Creating Graph')
    base.get_params()
    sess, queues, dbinterface, valid_targets_dict = base.test_from_params(**params)
    # start queue runners
    print('Starting queue runners')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    # get handles to network parts
    # data ops
    get_grids = valid_targets_dict['valid0']['targets']['full_grids']
    get_actions = valid_targets_dict['valid0']['targets']['actions']
    get_is_moving = valid_targets_dict['valid0']['targets']['is_moving']
    get_in_view = valid_targets_dict['valid0']['targets']['in_view']
    get_true_velocity = valid_targets_dict['valid0']['targets']['next_velocity']
    # run model ops
    predict_velocity = valid_targets_dict['valid0']['targets']['prediction']
    # placeholders
    grid_placeholder = valid_targets_dict['valid0']['targets']['grid_placeholder']
    # unroll across time
    print('Starting prediction')
    grids, actions, is_moving, in_view = sess.run([get_grids, get_actions, get_is_moving, get_in_view])
    grids = np.squeeze(grids)
    actions = np.squeeze(actions)
    is_moving = np.squeeze(is_moving)
    in_view = np.squeeze(in_view)
