'''
A simple test of tfutils, just to make training run. Currently, validation's a bit silly.
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
import curiosity.models.future_pred_symmetric_coupled_with_below as modelsource
from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)



CODE_ROOT = os.environ['CODE_ROOT']
cfgfile = os.path.join(CODE_ROOT, 
                       'curiosity/curiosity/configs/future_test_config_b.cfg')
cfg = postprocess_config(json.load(open(cfgfile)))



DATA_PATH = '/media/data2/one_world_dataset/dataset.lmdb'
VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/dataset8.lmdb'
BATCH_SIZE = 128
N = 2048000
NUM_BATCHES_PER_EPOCH = N // BATCH_SIZE
IMAGE_SIZE_CROP = 256
seed = 0

rng = np.random.RandomState(seed=seed)


params = {
	'save_params' : {
	    'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'future_pred_symmetric',
        'exp_id': 'test12_sepval',
        'save_valid_freq': 3000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 3000
	},

	'model_params' : {
		'func' : modelsource.model_tfutils_fpd_compatible,
		'rng' : None,
		'cfg' : cfg,
		'slippage' : 0
	},

	'train_params': {
        'data_params': {
            'func': FuturePredictionData,
            'data_path': DATA_PATH,
            'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],
    	    'random_time': False,
            'min_time_difference': 1,
    	    'batch_size': 128
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE,
            'n_threads': 1,
            'seed': 0,
    	    'capacity': BATCH_SIZE * 100
        },
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf')
    },


    'loss_params': {
        'targets': [],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.loss_per_case_fn,
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
                'random_time': False,
                'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],  # size after cropping an image
            	'min_time_difference': 1,
            	'batch_size': 128,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': BATCH_SIZE,
                'n_threads': 1,
                'seed': 0,
            		'capacity': BATCH_SIZE * 100,
            },
	    'targets': {
                'func': modelsource.loss_agg_for_validation,
                'targets': [],
            },
	    'agg_func': utils.mean_dict,
            'num_steps': 1 # N_VAL // BATCH_SIZE + 1,
            #'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            #'online_agg_func': online_agg
        }
    }



}



if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
