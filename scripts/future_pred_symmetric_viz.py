'''
A simple algorithm for loading a previously saved model, then saving input and outputs to the database for visualization.
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

CODE_BASE = os.environ['CODE_BASE']




params = {
	'model_params' : {
		modelsource.model_tfutils
	}
	'load_params' : {
		'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'future_pred_symmetric',
        'exp_id': 'test1',
	}
	'save_params' : {
		'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'future_pred_symmetric',
		'exp_id': 'save_im_1',
        'save_intermediate_freq': 1,
        'save_to_gfs': ['predicted', 'current', 'future', 'action']
	}
    'validation_params': {
        'valid0': {
            'data_params': {
                'func': FuturePredictionData,
                'data_path': VALID_DATA_PATH,  # path to image database
                'random_time': False,
                'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],  # size after cropping an image
		'min_time_difference': 10,
		'batch_size': 256,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': BATCH_SIZE,
                'n_threads': 1,
                'seed': 0,
		'capacity': BATCH_SIZE * 100,
            },
	    'targets': {
                'func': modelsource.loss_per_case_fn,
                'target': None,
            },
	    'agg_func': utils.mean_dict,
            'num_steps': 1 # N_VAL // BATCH_SIZE + 1,
            #'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            #'online_agg_func': online_agg
        }
}