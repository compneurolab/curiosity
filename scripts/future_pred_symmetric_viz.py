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

VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/dataset8.lmdb'
BATCH_SIZE = 128
# N = 2048000
# NUM_BATCHES_PER_EPOCH = N // BATCH_SIZE
IMAGE_SIZE_CROP = 256
seed = 0



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



# CODE_BASE = os.environ['CODE_BASE']

def get_extraction_target(inputs, outputs, to_extract, **loss_params):
    """
    Example validation target function to use to provide targets for extracting features.
    This function also adds a standard "loss" target which you may or not may not want

    The to_extract argument must be a dictionary of the form
          {name_for_saving: name_of_actual_tensor, ...}
    where the "name_for_saving" is a human-friendly name you want to save extracted
    features under, and name_of_actual_tensor is a name of the tensor in the tensorflow
    graph outputing the features desired to be extracted.  To figure out what the names
    of the tensors you want to extract are "to_extract" argument,  uncomment the
    commented-out lines, which will print a list of all available tensor names.
    """
    names = [[x.name for x in op.values()] for op in tf.get_default_graph().get_operations()]
    for n in names:
        print(n)

    targets = {k: tf.get_default_graph().get_tensor_by_name(v) for k, v in to_extract.items()}
    # targets['loss'] = utils.get_loss(inputs, outputs, **loss_params)
    return targets


def get_current_predicted_future_action(inputs, outputs, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) and outputs field (with pairs of arguments -- assuming outputs is a dict of dicts)
    '''
    futures = inputs['future_images'][:num_to_save]
    predictions = outputs['pred']['pred0'][:num_to_save]
    actions = inputs['actions'][:num_to_save]
    currents = inputs['images'][:num_to_save]
    futures_through = outputs['future']['future0'][:num_to_save]
    futures = tf.cast(futures, tf.uint8)
    predictions = tf.cast(tf.multiply(predictions, 255), tf.uint8)
    currents = tf.cast(currents, tf.uint8)
    futures_through = tf.cast(tf.multiply(futures_through, 255), tf.uint8)
    return {'prediction' : predictions, 'future_images' : futures, 'current_images': currents, 'actions' : actions, 'futures_through' : futures_through}


params = {
	'model_params' : {
		'func' : modelsource.model_tfutils_fpd_compatible,
        'rng' : None,
        'cfg' : cfg,
        'slippage' : 0
	},
	'load_params' : {
		'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'future_pred_symmetric',
        'exp_id': 'test12_sepval'
	},
	'save_params' : {
		'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'symmetric_viz',
		'exp_id': 'save_im_5',
        'save_intermediate_freq': 1,
        'save_to_gfs': ['predicted', 'current', 'future', 'action']
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
                'func': get_current_predicted_future_action,
                'targets' : []
            },
        'agg_func' : lambda x : {},
	    # 'agg_func': utils.mean_dict,
        'num_steps': 10 # N_VAL // BATCH_SIZE + 1,
            #'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            #'online_agg_func': online_agg
        }
    }
}


if __name__ == '__main__':
    base.get_params()
    base.test_from_params(**params)
