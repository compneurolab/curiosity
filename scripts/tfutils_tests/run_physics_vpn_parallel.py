import numpy as np
import os
import tensorflow as tf
import sys
import json

from tfutils import base, data, model, optimizer, utils
from curiosity.data.threeworld_data import ThreeWorldDataProvider
import curiosity.models.physics_vpn as modelsource
from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)

conf = 'cluster'

if conf is 'cluster':
    BASE_DIR = '/mnt/fs0/datasets/two_world_dataset'
    CACHE_DIR = '/mnt/fs0/mrowca/tfutils'
    HOST = 'localhost'
else:
    BASE_DIR = '/media/data2/new_dataset/'
    CACHE_DIR = '/media/data/mrowca/tfutils'
    HOST = 'localhost'

DATA_PATH = os.path.join(BASE_DIR, 'new_tfdata')
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, 'new_tfvaldata')
NORM_PATH = os.path.join(BASE_DIR, 'stats.pkl')

INPUT_BATCH_SIZE = 256
N_GPUS = 4
OUTPUT_BATCH_SIZE = 7 * N_GPUS
N = 2048000
NUM_BATCHES_PER_EPOCH = N // OUTPUT_BATCH_SIZE
IMAGE_SIZE_CROP = 256
TIME_DIFFERENCE = 1
SEQUENCE_LENGTH = 12
GAUSSIAN = None #['actions', 'poses']
SEGMENTATION = ['actions', 'positions']
RESIZE = {'images': [28, 64], 'objects': [28, 64]}
RANDOM_SKIP = None
USE_VALIDATION = True

seed = 0
exp_id = 'test30'

rng = np.random.RandomState(seed=seed)

def get_debug_info(inputs, outputs, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) 
    and outputs field (with pairs of arguments -- assuming outputs 
    is a dict of dicts)
    '''
    # ground truth images
    images = tf.stack(inputs['images'][:num_to_save])
    shape = images.get_shape().as_list()
    images = tf.reshape(images, [shape[0]*shape[1]] + shape[2:])
    images = tf.cast(images, tf.uint8)

    # ground truth actions
    actions = tf.stack(inputs['actions'][:num_to_save])
    shape = actions.get_shape().as_list()
    actions = tf.reshape(actions, [shape[0]*shape[1]] + shape[2:])

    # ground truth positions
    gt_pos = tf.stack(outputs['positions'][:num_to_save])
    shape = gt_pos.get_shape().as_list()
    gt_pos = tf.reshape(gt_pos, [shape[0]*shape[1]] + shape[2:])
    gt_pos = tf.cast(gt_pos, tf.uint8)
    gt_pos = tf.squeeze(gt_pos) * 255

    # predicted rgb image
    rgb = tf.stack(outputs['rgb'][:num_to_save])
    rgb = tf.nn.softmax(rgb)
    # maximum dimension that tf.argmax can handle is 5, so unstack here
    shape = rgb.get_shape().as_list()
    rgb = tf.reshape(rgb, [shape[0]*shape[1]] + shape[2:])
    rgb = tf.unstack(rgb)
    for i, r in enumerate(rgb):
        rgb[i] = tf.argmax(r, axis=tf.rank(r) - 1)
    rgb = tf.stack(rgb)
    rgb = tf.cast(rgb, tf.uint8)

    # predicted pos image
    pos = tf.stack(outputs['pos'][:num_to_save])
    pos = tf.nn.softmax(pos)
    shape = pos.get_shape().as_list()
    pos = tf.reshape(pos, [shape[0]*shape[1]] + shape[2:])
    pos = tf.argmax(pos, axis=tf.rank(pos) - 1)
    pos = tf.cast(pos, tf.uint8) * 255

    # return dict
    retval = {'img': images, 
            'pos': gt_pos,
            'act': actions,
            'pred_img': rgb, 
            'pred_pos': pos,}
    return retval

def keep_all(step_results):
    return step_results[0]

params = {
    'save_params' : {
        'host': HOST,
        'port': 27017,
        'dbname': 'tests',
        'collname': 'new_data',
        'exp_id': exp_id,
        'save_valid_freq': 500,
        'save_filters_freq': 50000,
        'cache_filters_freq': 2000,
        'save_metrics_freq': 50,
        'save_initial_filters' : False,
        'save_to_gfs': ['img', 'pos', 'act', 'pred_img', 'pred_pos'],
        'cache_dir': CACHE_DIR,
    },

    'load_params': {
        'host': HOST,
        # 'port': 31001,
        # 'dbname': 'alexnet-test'
        # 'collname': 'alexnet',
        # 'exp_id': 'trainval0',
        'port': 27017,
        'dbname': 'tests',
        'collname': 'new_data',
        #'exp_id': 'trainval0',
        'exp_id': exp_id,
        #'exp_id': 'trainval2', # using screen?
        'do_restore': True,
        'load_query': None
    },

    'model_params' : {
	'func' : modelsource.parallel_model,
        'batch_size': OUTPUT_BATCH_SIZE,
        'gaussian': GAUSSIAN,
        'segmentation': SEGMENTATION,
        'stats_file': NORM_PATH,
        'encoder_depth': 2,
        'decoder_depth': 4,
        'n_gpus': N_GPUS,
        #'normalization_method': {'images': 'standard', 'actions': 'minmax'},
    },

    'train_params': {
        'validate_first': True, #False,
        #'targets': {
        #    'func': modelsource.get_accuracy
        #},

        'data_params': {
            'func': ThreeWorldDataProvider,
            #'file_pattern': 'TABLE_CONTROLLED:DROP:FAST_PUSH:*.tfrecords',
            'data_path': DATA_PATH,
            'sources': ['images', 'actions', 'objects', 'object_data'],
            'n_threads': 4,
            'batch_size': INPUT_BATCH_SIZE,
            'delta_time': TIME_DIFFERENCE,
            'sequence_len': SEQUENCE_LENGTH,
            'output_format': 'sequence',
            'filters': ['is_not_teleporting', 'is_object_there'],
            'gaussian': GAUSSIAN,
            'max_random_skip': RANDOM_SKIP,
            'resize': RESIZE,
        },

        'queue_params': {
            'queue_type': 'random',
            'batch_size': OUTPUT_BATCH_SIZE,
            'seed': seed,
    	    'capacity': 11*INPUT_BATCH_SIZE,
            'min_after_dequeue': 10*INPUT_BATCH_SIZE,
        },
        
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf')
    },

    'loss_params': {
        'targets': ['images', 'actions'],
        'agg_func': modelsource.parallel_reduce_mean,
        'loss_per_case_func': modelsource.parallel_softmax_cross_entropy_loss,
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 0.0001,
        'decay_rate': 0.95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'optimizer_params': {
        'func': modelsource.ParallelClipOptimizer,
        'optimizer_class': tf.train.RMSPropOptimizer,
        'clip': True,
        'momentum': .9
    }
}

if USE_VALIDATION:
    params['validation_params'] = {
        'valid0': {
            'data_params': {
                'func': ThreeWorldDataProvider,
                #'file_pattern': 'TABLE_CONTROLLED:DROP:FAST_PUSH:*.tfrecords',
                'data_path': DATA_PATH,
                'sources': ['images', 'actions', 'objects', 'object_data'],
                'n_threads': 4,
                'batch_size': INPUT_BATCH_SIZE,
                'delta_time': TIME_DIFFERENCE,
                'sequence_len': SEQUENCE_LENGTH,
                'output_format': 'sequence',
                'filters': ['is_not_teleporting', 'is_object_there'],
                'gaussian': GAUSSIAN,
                'max_random_skip': RANDOM_SKIP,
                'resize': RESIZE,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': OUTPUT_BATCH_SIZE,
                'seed': seed,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            },
            'targets': {
                'func': get_debug_info,
                'targets' : [],
                'num_to_save' : 5
            },
        'agg_func' : keep_all,
        #'agg_func': utils.mean_dict,
        'num_steps': 10 # N_VAL // BATCH_SIZE + 1,
        #'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        #'online_agg_func': online_agg
        }
    }

if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
