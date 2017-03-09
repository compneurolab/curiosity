'''
A simple explicit position prediction script, this time filtering out teleport, look, and moving closer.
'''

uniform_width = 400

cfg_50pts = {
	'hidden_depth' : 6,
	'hidden' : {
		1 : {'num_features' : uniform_width},
		2 : {'num_features' : uniform_width},
		3 : {'num_features' : uniform_width},
		4 : {'num_features' : uniform_width},
		5 : {'num_features' : uniform_width},
		6 : {'num_features' : uniform_width}
	}
}

cfg_linear = {
    'hidden_depth' : 0
}

big_uniform_width = 2000

cfg_big = {
    'hidden_depth' : 8,
    'hidden' : {
        1 : {'num_features' : big_uniform_width},
        2 : {'num_features' : big_uniform_width},
        3 : {'num_features' : big_uniform_width},
        4 : {'num_features' : big_uniform_width},
        5 : {'num_features' : big_uniform_width},
        6 : {'num_features' : big_uniform_width},
        7 : {'num_features' : big_uniform_width},
        8 : {'num_features' : big_uniform_width}
    }
}
import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import json

from tfutils import base, data, model, optimizer, utils
from curiosity.data.explicit_positions import PositionPredictionData 
import curiosity.models.explicit_position_models as modelsource

DATA_PATH = '/media/data2/one_world_dataset/tfdata'
VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/tfvaldata'
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
DISCRETE_THRESHOLD = .1
MAX_NUM_OBJECTS = 100
MAX_NUM_ACTIONS = 10
OUTPUT_NUM_OBJ = 10
N = 2048000
NUM_BATCHES_PER_EPOCH = N // MODEL_BATCH_SIZE
seed = 0
T_in = 3
T_out = 3
SEQ_LEN = T_in + T_out

#TODO should keep loss for all!
def append_every_kth(x, y, step, k):
    if x is None:
        x = []
    if step % k == 0:
        x.append(y)
    return x



def get_ins_and_outs(inputs, outputs, num_to_save = 1, t_in = 3, t_out = 3, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) and outputs field (with pairs of arguments -- assuming outputs is a dict of dicts)
    '''
    num_points = inputs['positions'].get_shape().as_list()[1] / (3 * (t_in + t_out))
    input_pos = inputs['positions'][:num_to_save, : num_points * 3 * t_in]
    output_pos = outputs['tv'][:num_to_save]
    pred = outputs['pred'][:num_to_save]
    retval = {'inpos' : input_pos, 'outpos' : output_pos, 'pred' : pred}
    retval.update(get_loss_for_val(inputs, outputs, ** loss_params))
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
   	return {'loss' : modelsource.l2_loss_fn(outputs, inputs['positions'])}



params = {
	'save_params' : {
	    'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'positions',
        'exp_id': 'lin_scale5',
        'save_valid_freq': 2000,
        'save_filters_freq': 1000000,
        'cache_filters_freq': 50000,
        'save_initial_filters' : False,
        'save_to_gfs': ['inpos', 'outpos', 'pred'],
        'cache_dir' : '/media/data/nhaber',
	},

	'model_params' : {
		'func' : modelsource.position_only_mlp,
		'cfg' : cfg_linear,
        'T_in' : T_in,
        'T_out' : T_out,
        'num_points' : OUTPUT_NUM_OBJ,
        'stddev' : .01,
	},

	'train_params': {
        'data_params': {
            'func': PositionPredictionData,
            'num_timesteps' : SEQ_LEN,
            'max_num_objects' : MAX_NUM_OBJECTS,
            'max_num_actions' : MAX_NUM_ACTIONS,
            'output_num_objects' : OUTPUT_NUM_OBJ,
            'data_path': DATA_PATH,
    	    'batch_size': DATA_BATCH_SIZE,
            'n_threads' : 4,
            'random' : True,
            'random_seed' : 0,
            'positions_only' : True,
            'filters' : ['act_mask_1'],
            'manual_coord_scaling' : [10., 1., 10.]
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': MODEL_BATCH_SIZE,
            'seed': 0,
    	    'capacity': MODEL_BATCH_SIZE * 12,
        },
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf'),
    },


    'loss_params': {
        'targets': ['positions'],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.l2_diff_loss_fn,
		# 'loss_func_kwargs' : {
		# 	'num_channels' : 3,
  #           'threshold' : DISCRETE_THRESHOLD
		# },
		'loss_func_kwargs' : {'t_in' : T_in, 't_out' : T_out},
		'loss_per_case_func_params' : {}
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-3,
        'decay_rate': 0.95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'validation_params': {
        'valid0': {
            'data_params': {
	            'func': PositionPredictionData,
	            'num_timesteps' : SEQ_LEN,
	            'max_num_objects' : MAX_NUM_OBJECTS,
	            'max_num_actions' : MAX_NUM_ACTIONS,
	            'output_num_objects' : OUTPUT_NUM_OBJ,
	            'data_path': VALIDATION_DATA_PATH,
	    	    'batch_size': DATA_BATCH_SIZE,
	            'n_threads' : 1,
	            'random' : True,
	            'random_seed' : 0,
	            'positions_only' : True,
                'filters' : ['act_mask_1'],
                'manual_coord_scaling' : [10., 1., 10.]
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MODEL_BATCH_SIZE,
                'seed': 0,
              'capacity': MODEL_BATCH_SIZE * 1,
                # 'n_threads' : 4

            },
        'targets': {
                'func': get_ins_and_outs,
                'targets' : [],
                'num_to_save' : 1,
                't_in' : T_in,
                't_out' : T_out
            },
        'agg_func' : mean_losses_keep_rest,
        'online_agg_func' : lambda x, y, step : append_every_kth(x, y, step, 10),
        'num_steps': 100
        }
    }



}


if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)





