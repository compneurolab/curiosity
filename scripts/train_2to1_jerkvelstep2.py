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

DATA_PATH = '/mnt/fs1/datasets/six_world_dataset/new_tfdata'
VALDATA_PATH = '/mnt/fs1/datasets/six_world_dataset/new_tfvaldata'
#DATA_PATH = '/data/two_world_dataset/new_tfdata'
#VALDATA_PATH = '/data/two_world_dataset/new_tfvaldata'

N_GPUS = 4
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 64
TIME_SEEN = 3
SHORT_LEN = TIME_SEEN
LONG_LEN = 4
MIN_LEN = 4
CACHE_DIR = '/mnt/fs0/mrowca/cache3/'
NUM_BATCHES_PER_EPOCH = 4000 * 256 / MODEL_BATCH_SIZE
STATS_FILE = '/mnt/fs1/datasets/six_world_dataset/new_stats/stats_std.pkl'
BIN_PATH = '/mnt/fs1/datasets/six_world_dataset/'
BIN_FILE = '/mnt/fs1/datasets/six_world_dataset/bin_data_file.pkl'
IMG_HEIGHT = 128
IMG_WIDTH = 170
SCALE_DOWN_HEIGHT = 32
SCALE_DOWN_WIDTH = 43
L2_COEF = 200.
EXP_ID = ['vel_model_act_vel', 
'vel_model_act_vel_seg',
'vel_model_vel', 
'vel_model_vel_seg']
#EXP_ID = ['res_jerk_eps', 'map_jerk_eps', 'sym_jerk_eps', 'bypass_jerk_eps']
LRS = [0.001, 0.001, 0.001, 0.001]
n_classes = 768
buckets = 255
CFG = [ modelsource.cfg_mom_concat(n_classes, use_cond=True, method='concat'),
        modelsource.cfg_mom_concat(n_classes, use_cond=True, method='concat'), 
        modelsource.cfg_mom_concat(n_classes, use_cond=False, method='concat'), 
        modelsource.cfg_mom_concat(n_classes, use_cond=False, method='concat')]
USE_VEL = [True, True, True, True]
USE_SEG = [False, True, False, True]
CACHE_DIRS = [CACHE_DIR + str(d) for d in range(4)]
SEED = 0

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

SAVE_TO_GFS = ['object_data_future', 'pred_next_vel_1', 'object_data_seen_1d', 'reference_ids', 'master_filter', 'jerk_map', 'vels']

def grab_all(inputs, outputs, bin_file = BIN_FILE, 
        num_to_save = 1, gpu_id = 0, **garbage_params):
    retval = {}
    batch_size = outputs['pred_next_vel_1'].get_shape().as_list()[0]
    retval['loss'] = modelsource.softmax_cross_entropy_loss_vel( 
            outputs, gpu_id=gpu_id, segmented_jerk=False, use_current_vel_loss=False,
            buckets=buckets)
    for k in SAVE_TO_GFS:
        if k != 'reference_ids':
            if k in ['pred_vel_1', 'pred_next_vel_1']:
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
    'func' : modelsource.mom_model_step2,
    'cfg' : CFG[0],
    'time_seen' : TIME_SEEN,
    'normalization_method' : {
        #'object_data' : 'screen_normalize', 
        'actions' : 'minmax'},
    'stats_file' : STATS_FILE,
    'image_height' : IMG_HEIGHT,
    'image_width' : IMG_WIDTH,
    #'scale_down_height' : SCALE_DOWN_HEIGHT,
    #'scale_down_width' : SCALE_DOWN_WIDTH,
    'add_depth_gaussian' : False,
    'use_vel': False,
    'use_segmentation': False,
    'include_pose' : False,
    #'num_classes': 60.,
    'gpu_id' : 0,
}] * N_GPUS

loss_params = [{
    'targets' : [],
    'agg_func' : modelsource.parallel_reduce_mean,
    'loss_per_case_func' : modelsource.softmax_cross_entropy_loss_vel,
    'loss_per_case_func_params' : {'_outputs': 'outputs', '_targets_$all': 'inputs'},
    'loss_func_kwargs' : {'bin_data_file': BIN_FILE, 'gpu_id': 0, 
        'buckets': buckets, 'segmented_jerk': False, 'use_current_vel_loss': False}, 
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
    'optimizer_class': tf.train.AdamOptimizer,
    'clip': True,
    'gpu_offset': 0,
    #'momentum': .9,
    'beta1': 0.95,
    'beta2': 0.9995,
}] * N_GPUS

validation_params = [{
    'valid0' : {
        'data_params' : {
            'func' : ShortLongSequenceDataProvider,
            'data_path' : VALDATA_PATH,
            'short_sources' : [], #'depths2', 'normals2', 'images'
            'long_sources' : ['depths', 
                'jerks', 'accs', 'vels', 'accs_curr', 'vels_curr',
                'actions', 'objects', 'object_data', 'reference_ids'],
            'short_len' : SHORT_LEN,
            'long_len' : LONG_LEN,
            'min_len' : MIN_LEN,
            'filters' : ['is_not_teleporting', 'is_object_in_view'],
            'shuffle' : True,
            'shuffle_seed' : SEED,
            'n_threads' : 1,
            'batch_size' : DATA_BATCH_SIZE,
            'file_grab_func' : table_norot_grab_func,
           # 'is_there_subsetting_rule' : 'just_first',
            'is_in_view_subsetting_rule' : 'both_there',
            },
        'queue_params' : {
            'queue_type' : 'random',
            'batch_size' : MODEL_BATCH_SIZE,
            'seed' : SEED,
            'capacity' : MODEL_BATCH_SIZE * 20,
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
        'num_steps' : 20,
    },
}] * N_GPUS

train_params =  {
    'validate_first': False,
    'data_params' : {
        'func' : ShortLongSequenceDataProvider,
        'data_path' : DATA_PATH,
        'short_sources' : [], #'depths2', 'normals2', 'images' 
        'long_sources' : ['depths',
            'jerks', 'accs', 'vels', 'accs_curr', 'vels_curr',            
            'actions', 'objects', 'object_data', 'reference_ids'],
        'short_len' : SHORT_LEN,
        'long_len' : LONG_LEN,
        'min_len' : MIN_LEN,
        'filters' : ['is_not_teleporting', 'is_object_in_view'],
        'shuffle' : True,
        'shuffle_seed' : SEED,
        'n_threads' : 4,
        'batch_size' : DATA_BATCH_SIZE,
        'file_grab_func' : table_norot_grab_func,
        #'is_there_subsetting_rule' : 'just_first',
        'is_in_view_subsetting_rule' : 'both_there',
    },
        
    'queue_params' : {
        'queue_type' : 'random',
        'batch_size' : MODEL_BATCH_SIZE,
        'seed' : SEED,
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

    save_params[i]['cache_dir'] = CACHE_DIRS[i]
    loss_params[i]['loss_func_kwargs']['gpu_id'] = i
    #loss_params[i]['loss_func_kwargs']['bin_data_file'] = BIN_PATH + EXP_ID[i] + '.pkl'
    model_params[i]['gpu_id'] = i
    optimizer_params[i]['gpu_offset'] = i
    validation_params[i]['valid0']['targets']['gpu_id'] = i
    #validation_params[i]['valid0']['targets']['bin_file'] = BIN_PATH + EXP_ID[i] + '.pkl'
    model_params[i]['cfg'] = CFG[i]
    model_params[i]['use_vel'] = USE_VEL[i]
    model_params[i]['use_segmentation'] = USE_SEG[i]
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

def int_shape(x):
        return list(map(int, x.get_shape()))

def sample_from_discretized_mix_logistic(l, nr_mix, buckets = 255.0):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(
        logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(tf.reduce_sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, 4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, 4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], 0.), buckets) #-1.), 1.)
    x1 = tf.minimum(tf.maximum(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, 0.), buckets) #-1.), 1.)
    x2 = tf.minimum(tf.maximum(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, 0.), buckets) #-1.), 1.)
    return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], 3)


if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
