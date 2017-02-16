'''
Running a model that's meant to do timestep 1 evolution in the hidden layers.
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
import curiosity.models.asymmetric_modularized as modelsource
from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)


cfg_alexy = {
	'encode_depth' : 5,
	'decode_depth' : 5,
	'hidden_depth' : 3,
	'encode' : {
		1 : {'conv' : {'filter_size' : 11, 'stride' : 4, 'num_filters' : 96}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 384}},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 384}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256}}, # size 256 image, this leads to 16 * 16 * 256 = 65,536 neurons. Sad!
	},



	'decode' : {
		0 : {'size' : 16, 'num_filters' : 256},
		1 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 384},
		2 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 384},
		3 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 256},
		4 : {'filter_size' : 5, 'size' : 32, 'num_filters' : 96},
		5 : {'filter_size' : 11, 'size' : 256, 'num_filters' : 3}
	},

	'hidden' : {
		1 : {'num_features' : 256},
		2 : {'num_features' : 256},
		3 : {'num_features' : 65536}
	}

}

cfg_alexy_small = {
	'encode_depth' : 5,
	'decode_depth' : 5,
	'hidden_depth' : 4,
	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 96}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 2, 'num_filters' : 96}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 96}},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32}}, # Size 16 image, giving 16 * 16 * 32 = 8192 neurons. Let's try it!
	},

	'decode' : {
		0 : {'size' : 16, 'num_filters' : 4},
		1 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 64},
		2 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 32},
		3 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 32},
		4 : {'filter_size' : 5, 'size' : 64, 'num_filters' : 32},
		5 : {'filter_size' : 11, 'size' : 256, 'num_filters' : 3}
	},

	'hidden' : {
		0 : {'num_features' : 1024},
		1 : {'num_features' : 1024},
		2 : {'num_features' : 1024},
		3 : {'num_features' : 1024},
		4 : {'num_features' : 1024}
	}

}

cfg = cfg_alexy_small

DATA_PATH = '/media/data2/one_world_dataset/dataset_images_parsed_actions.tfrecords'
VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/dataset_images_parsed_actions8.tfrecords'
BATCH_SIZE = 128
DISCRETE_THRESHOLD = .1
N = 2048000
NUM_BATCHES_PER_EPOCH = N // BATCH_SIZE
IMAGE_SIZE_CROP = 256
seed = 0
T_in = 3
T_out = 3
SEQ_LEN = T_in + T_out


def get_current_predicted_future_action(inputs, outputs, num_channels, threshold = None, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) and outputs field (with pairs of arguments -- assuming outputs is a dict of dicts)
    '''
    images = inputs['images'][:num_to_save]
    predictions = outputs['pred'][:num_to_save]
    actions = inputs['parsed_actions'][:num_to_save]
    retval = {'pred' : predictions, 'img' : images, 'act' : actions}
    retval.update(get_loss_for_val(inputs, outputs, num_channels = num_channels, threshold = threshold, **loss_params))
    #compute the trueval and save, so we don't have the sort of debacle like before
    future_images = tf.cast(outputs['tv'][:num_to_save], 'float32')
    assert threshold is not None
    T_in = int((images.get_shape().as_list()[-1] -  predictions.get_shape().as_list()[-1]) / num_channels)
    original_image = images[:, :, :, (T_in - 1) * num_channels: T_in * num_channels]
    original_image = tf.cast(original_image, 'float32')
    diffs = modelsource.compute_diffs_timestep_1(original_image, future_images, num_channels = num_channels)
    #just measure some absolute change relative to a threshold
    diffs = tf.abs(diffs / 255.) - threshold
    tv = tf.cast(tf.ceil(diffs), 'uint8')
    tv = tf.one_hot(tv, depth = 2)
    retval['tv'] = tv
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


def get_loss_for_val(inputs, outputs, num_channels = 3, threshold = None, **loss_params):
   	return {'loss' : modelsource.something_or_nothing_loss_fn(outputs, inputs['images'], num_channels = num_channels, threshold = threshold)}

def get_ids_target(inputs, outputs, *ttarg_params):
    return {'ids' : inputs['id']}



params = {
	'save_params' : {
	    'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'asymmetric',
        'exp_id': '33_t1',
        'save_valid_freq': 2000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 2000,
        'save_initial_filters' : False,
        'save_to_gfs': ['pred', 'img', 'act', 'tv', 'ids'],
        'cache_dir' : '/media/data/nhaber',
	},

	'model_params' : {
		'func' : modelsource.timestep_1_mlp_model,
		'rng' : None,
		'cfg' : cfg,
		'slippage' : 0,
        'T_in' : T_in,
        'T_out' : T_out
	},

	'train_params': {
        'data_params': {
            'func': FuturePredictionData,
            'data_path': DATA_PATH,
            # 'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],
            'output_format' : {'images' : 'sequence', 'actions' : 'sequence'},
            'min_time_difference': SEQ_LEN,
    	    'batch_size': BATCH_SIZE,
            'n_threads' : 2,
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE,
            'seed': 0,
    	    'capacity': BATCH_SIZE * 60,
            # 'n_threads' : 4
        },
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf'),
        'targets' : {'func' : get_ids_target}
    },


    'loss_params': {
        'targets': ['images'],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.something_or_nothing_loss_fn,
		'loss_func_kwargs' : {
			'num_channels' : 3,
            'threshold' : DISCRETE_THRESHOLD
		},
		'loss_per_case_func_params' : {}
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 0.001,
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
                'min_time_difference': SEQ_LEN,
                'batch_size': BATCH_SIZE,
                'n_threads' : 2,
                'output_format' : {'images' : 'sequence', 'actions' : 'sequence'}
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': BATCH_SIZE,
                'seed': 0,
              'capacity': BATCH_SIZE * 2,
                # 'n_threads' : 4

            },
        'targets': {
                'func': get_current_predicted_future_action,
                'targets' : [],
                'num_to_save' : 1,
                'num_channels' : 3,
                'threshold' : DISCRETE_THRESHOLD
            },
        'agg_func' : mean_losses_keep_rest,
        'num_steps': 10
        }
    }



}


if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)




