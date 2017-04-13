import numpy as np
import os
import tensorflow as tf
import sys
import json

from tfutils import base, data, model, optimizer, utils
from curiosity.data.threeworld_data import ThreeWorldDataProvider
import curiosity.models.vpn as modelsource
from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)

#cfgfile = os.path.join('/home/mrowca/workspace/', 
#                       'curiosity/curiosity/configs/action_config2.cfg')
#cfg = postprocess_config(json.load(open(cfgfile)))

DATA_PATH = '/media/data2/new_dataset/new_tfdata'
VALIDATION_DATA_PATH = '/media/data2/new_dataset/new_tfdata'
#STATS_FILE = '/media/data/one_world_dataset/dataset_stats.pkl'
NORM_PATH = '/media/data2/new_dataset/stats.pkl'

INPUT_BATCH_SIZE = 256
OUTPUT_BATCH_SIZE = 16
N = 2048000
NUM_BATCHES_PER_EPOCH = N // OUTPUT_BATCH_SIZE
IMAGE_SIZE_CROP = 256
TIME_DIFFERENCE = 1
SEQUENCE_LENGTH = 10
GAUSSIAN = None #['actions', 'poses']
RESIZE = {'images': [14, 32]}
RANDOM_SKIP = None
seed = 0
exp_id = 'test4'

rng = np.random.RandomState(seed=seed)

def get_debug_info(inputs, outputs, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) 
    and outputs field (with pairs of arguments -- assuming outputs 
    is a dict of dicts)
    '''
    images = inputs['images'][:num_to_save]
    images = tf.cast(images, tf.uint8)

    #actions = outputs['actions'][:num_to_save]
    
    retval = {'img' : images}
    return retval

def keep_all(step_results):
    return step_results[0]

params = {
    'save_params' : {
        'host': 'localhost',
        'port': 29101,
        'dbname': 'tests',
        'collname': 'new_data',
        'exp_id': exp_id,
        'save_valid_freq': 500,
        'save_filters_freq': 50000,
        'cache_filters_freq': 2000,
        'save_metrics_freq': 50,
        'save_initial_filters' : False,
        'save_to_gfs': ['act', 'img'],
        'cache_dir': '/media/data/mrowca/tfutils'
    },

    'load_params': {
        'host': 'localhost',
        # 'port': 31001,
        # 'dbname': 'alexnet-test'
        # 'collname': 'alexnet',
        # 'exp_id': 'trainval0',
        'port': 29101,
        'dbname': 'tests',
        'collname': 'new_data',
        #'exp_id': 'trainval0',
        'exp_id': exp_id,
        #'exp_id': 'trainval2', # using screen?
        'do_restore': False,
        'load_query': None
    },

    'model_params' : {
	'func' : modelsource.model,
        'batch_size': OUTPUT_BATCH_SIZE,
        'gaussian': GAUSSIAN,
        'stats_file': NORM_PATH,
        'encoder_depth': 4,
        'decoder_depth': 6,
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
            'sources': ['images'],
            'n_threads': 1,
            'batch_size': INPUT_BATCH_SIZE,
            'delta_time': TIME_DIFFERENCE,
            'sequence_len': SEQUENCE_LENGTH,
            'output_format': 'sequence',
            'filters': ['is_not_teleporting'],
            'gaussian': GAUSSIAN,
            'max_random_skip': RANDOM_SKIP,
            'resize': RESIZE,
        },

        'queue_params': {
            'queue_type': 'fifo',
            'batch_size': OUTPUT_BATCH_SIZE,
            'seed': 0,
    	    'capacity': None,
            'min_after_dequeue': None,
        },
        
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf')
    },

    'loss_params': {
        'targets': ['images'],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.softmax_cross_entropy_loss,
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 0.0001,
        'decay_rate': 0.95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.RMSPropOptimizer,
        'clip': True,
        'momentum': .9
    },

    'validation_params': {
        'valid0': {
            'data_params': {
                'func': ThreeWorldDataProvider,
                #'file_pattern': 'TABLE_CONTROLLED:DROP:FAST_PUSH:*.tfrecords',
                'data_path': DATA_PATH,
                'sources': ['images'],
                'n_threads': 1,
                'batch_size': INPUT_BATCH_SIZE,
                'delta_time': TIME_DIFFERENCE,
                'sequence_len': SEQUENCE_LENGTH,
                'output_format': 'sequence',
                'filters': ['is_not_teleporting'],
                'gaussian': GAUSSIAN,
                'max_random_skip': RANDOM_SKIP,
                'resize': RESIZE,
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': OUTPUT_BATCH_SIZE,
                'seed': 0,
                'capacity': None,
                'min_after_dequeue': None,
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
}


if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
