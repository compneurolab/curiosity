'''
A simple test of tfutils, just to make training run. Currently, validation's a bit silly.
'''

import numpy as np
import os
import tensorflow as tf
import sys
import json

from tfutils import base, data, model, optimizer, utils
from curiosity.data.images_futures_and_actions import FuturePredictionData 
import curiosity.models.tf_action_pred_sequence as modelsource
from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)

cfgfile = os.path.join('/home/mrowca/workspace/', 
                       'curiosity/curiosity/configs/action_config2.cfg')
cfg = postprocess_config(json.load(open(cfgfile)))

DATA_PATH = '/media/data2/one_world_dataset/tfdata'
VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/tfvaldata'
STATS_FILE = '/media/data/one_world_dataset/dataset_stats.pkl'

INPUT_BATCH_SIZE = 256
OUTPUT_BATCH_SIZE = 128
N = 2048000
NUM_BATCHES_PER_EPOCH = N // OUTPUT_BATCH_SIZE
IMAGE_SIZE_CROP = 256
TIME_DIFFERENCE = 5
seed = 0
exp_id = 'test68'

rng = np.random.RandomState(seed=seed)

def get_high_loss_inputs(inputs, outputs):
    return {'high_loss_actions': inputs['parsed_actions'], \
            'high_loss_norm': outputs['norm_actions']}

def get_current_predicted_future_action(inputs, outputs, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) 
    and outputs field (with pairs of arguments -- assuming outputs 
    is a dict of dicts)
    '''
    currents = inputs['images'][:num_to_save]
    currents = tf.cast(currents, tf.uint8)

    futures = inputs['future_images'][:num_to_save]
    futures = tf.cast(futures, tf.uint8)

    actions = inputs['parsed_actions'][:num_to_save]
    predictions = outputs['pred'][:num_to_save]
    norm_actions = outputs['norm_actions'][:num_to_save]

    shape = actions.get_shape().as_list()
    norm = shape[0] * shape[1]
    loss = tf.nn.l2_loss(predictions - norm_actions) / norm

    retval = {'pred' : predictions, 'fut' : futures, \
              'cur': currents, 'act' : actions, \
              'norm': norm_actions, 'val_loss': loss}
    return retval


def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    for k in keys:
        plucked = [d[k] for d in step_results]
        if 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval


params = {
    'save_params' : {
        'host': 'localhost',
        'port': 27017,
        'dbname': 'acion_pred',
        'collname': 'action_pred_symmetric',
        'exp_id': exp_id,
        'save_valid_freq': 1000,
        'save_filters_freq': 50000,
        'cache_filters_freq': 2000,
        'save_metrics_freq': 100,
        'save_initial_filters' : False,
        'save_to_gfs': ['act', 'pred', 'fut', 'cur', 'norm', 'val_loss'],
        #'high_loss_actions', 'high_loss_norm'],
        'cache_dir': '/media/data/mrowca/tfutils'
    },

    'load_params': {
        'host': 'localhost',
        # 'port': 31001,
        # 'dbname': 'alexnet-test'
        # 'collname': 'alexnet',
        # 'exp_id': 'trainval0',
        'port': 27017,
        'dbname': 'acion_pred',
        'collname': 'action_pred_symmetric',
        #'exp_id': 'trainval0',
        'exp_id': exp_id,
        #'exp_id': 'trainval2', # using screen?
        'do_restore': True,
        'load_query': None
    },

    'model_params' : {
	'func' : modelsource.actionPredModel,
	'rng' : None,
	'cfg' : cfg,
	'slippage' : 0,
        'min_time_difference': TIME_DIFFERENCE,
        'min_max_end' : False,
        'normalize_images_after_dequeue': True, 
        'diff_mode' : False,
        'n_channels': 3,
    },

    'train_params': {
        'validate_first': False,

        #'targets': {
        #    'func': get_high_loss_inputs,
        #},

        'data_params': {
            'func': FuturePredictionData,
            'data_path': DATA_PATH,
            #'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],
            'min_time_difference': TIME_DIFFERENCE,
            'output_format': {'images': 'sequence', 'actions': 'sequence'},
            'use_object_ids': False,
            'normalize_actions': 'custom',
            #'normalize_images': 'standard' BLOWS UP MEMORY!,
            'stats_file': STATS_FILE,
            'action_matrix_radius': None,
    	    'batch_size': INPUT_BATCH_SIZE,
            'shuffle': True,
            'shuffle_seed': 0,
            'n_threads': 4,
        },

        'queue_params': {
            'queue_type': 'random',
            'batch_size': OUTPUT_BATCH_SIZE,
            'seed': 0,
    	    'capacity': OUTPUT_BATCH_SIZE * 20
        },
        
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf')
    },

    'loss_params': {
        'targets': ['parsed_actions'],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.weighted_l2_action_loss,
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 0.01,
        'decay_rate': 0.95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.MomentumOptimizer,
        'clip': True,
        'momentum': .9
    },

    'validation_params': {
        'valid0': {
            'data_params': {
                'func': FuturePredictionData,
                'data_path': VALIDATION_DATA_PATH,  # path to image database
                #'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP]
                'output_format': {'images': 'sequence', 'actions': 'sequence'},
                'normalize_actions': 'custom',
                #'normalize_images': 'standard', BLOWS UP MEMORY
                'stats_file': STATS_FILE,
                'use_object_ids': False,
                'action_matrix_radius': None,
                'min_time_difference': TIME_DIFFERENCE,
                'batch_size': INPUT_BATCH_SIZE,
                'shuffle': True,
                'shuffle_seed': 0,
                'n_threads': 4,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': OUTPUT_BATCH_SIZE,
                'seed': 0,
              'capacity': OUTPUT_BATCH_SIZE * 10,
            },
            'targets': {
                'func': get_current_predicted_future_action,
                'targets' : [],
                'num_to_save' : 1
            },
        'agg_func' : mean_losses_keep_rest,
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
