'''
Basic asymmetric network training script.
'''

import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import json

from tfutils import base, data, model, optimizer, utils
from curiosity.data.images_futures_and_actions import FuturePredictionData 
import curiosity.models.future_pred_asymmetric_with_bypass2 as modelsource
from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)


CODE_ROOT = os.environ['CODE_ROOT']
cfgfile = os.path.join(CODE_ROOT, 
                       'curiosity/curiosity/configs/asymmetric_test.cfg')
cfg = postprocess_config(json.load(open(cfgfile)))



DATA_PATH = '/media/data2/one_world_dataset/dataset_images_parsed_actions.tfrecords'
VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/dataset_images_parsed_actions8.tfrecords'
BATCH_SIZE = 128
N = 2048000
NUM_BATCHES_PER_EPOCH = N // BATCH_SIZE
IMAGE_SIZE_CROP = 256
seed = 0

def get_current_predicted_future_action(inputs, outputs, num_classes, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) and outputs field (with pairs of arguments -- assuming outputs is a dict of dicts)
    '''
    futures = inputs['future_images'][:num_to_save]
    predictions = outputs['pred'][:num_to_save]
    actions = inputs['parsed_actions'][:num_to_save]
    currents = inputs['images'][:num_to_save]
    futures = tf.cast(futures, tf.uint8)
    # predictions = tf.cast(tf.multiply(predictions, 255), tf.uint8)
    currents = tf.cast(currents, tf.uint8)
    diffs = tf.cast(futures, 'float32') - tf.cast(currents, 'float32')
    retval = {'pred' : predictions, 'fut' : futures, 'cur': currents, 'act' : actions, 'diff' : diffs}
    retval.update(get_loss_for_val(inputs, outputs, num_classes = num_classes, **loss_params))
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


def get_loss_for_val(inputs, outputs, num_classes = 1, **loss_params):
   	return {'loss' : modelsource.something_or_nothing_loss_fn(outputs, inputs['images'], inputs['future_images'])}





params = {
	'save_params' : {
	    'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'asymmetric',
        'exp_id': 'test1',
        'save_valid_freq': 3000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 3000,
        'save_initial_filters' : False,
        'save_to_gfs': ['act', 'pred', 'fut', 'cur', 'diff'],
        'cache_dir' : '/media/data/nhaber'
	},

	'model_params' : {
		'func' : modelsource.model_tfutils_fpd_compatible,
		'rng' : None,
		'cfg' : cfg,
		'slippage' : 0,
	},

	'train_params': {
        'data_params': {
            'func': FuturePredictionData,
            'data_path': DATA_PATH,
            # 'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],
            'min_time_difference': 4,
    	    'batch_size': 256,
            'n_threads' : 4
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE,
            'seed': 0,
    	    'capacity': BATCH_SIZE * 100,
            # 'n_threads' : 4
        },
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf')
    },


    'loss_params': {
        'targets': ['images', 'future_images'],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.something_or_nothing_loss_fn,
		# 'loss_func_kwargs' : {
		# 	'num_classes' : 1
		# }
		'loss_per_case_func_params' : {}
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 0.05,
        'decay_rate': 0.95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'validation_params': {
        'valid0': {
            'data_params': {
                'func': FuturePredictionData,
                'data_path': VALIDATION_DATA_PATH,  # path to image database
                # 'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],  # size after cropping an image
                'min_time_difference': 4,
                'batch_size': 128,
                'n_threads' : 4,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': BATCH_SIZE,
                'seed': 0,
              'capacity': BATCH_SIZE * 100,
                # 'n_threads' : 4

            },
        'targets': {
                'func': get_current_predicted_future_action,
                'targets' : [],
                'num_to_save' : 2,
                'num_classes' : 1
            },
        'agg_func' : mean_losses_keep_rest,
        'num_steps': 1 
        }
    }



}


if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)




