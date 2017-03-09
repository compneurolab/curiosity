'''
Now, time to do things with random skips
'''

cfg_naive = {
    'hidden_depth' : 2,
    'hidden' : {
        1 : {'num_features' : 100},
        2 : {'num_features' : 100},
        # 3 : {'num_features' : 10}
    }
}

cfg_lin = {
    'first_lin' : 27
}


import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')

from tfutils import base, data, model, optimizer, utils
from curiosity.data.explicit_positions import RandomParabolaRandomFutureTimeGenerator 
import curiosity.models.synthetic_explicit_position_models as modelsource

DATA_PATH = '/media/data2/one_world_dataset/tfdata'
VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/tfvaldata'
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
N = 2048000
NUM_BATCHES_PER_EPOCH = N // MODEL_BATCH_SIZE
seed = 0
T_in = 3
T_out = 3
# SKIP = 4
# SEQ_LEN = T_in + T_out + SKIP


#TODO should keep loss for all!
def append_every_kth(x, y, step, k):
    if x is None:
        x = []
    if step % k == 0:
        x.append(y)
    return x



def get_ins_and_outs(inputs, outputs, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) and outputs field (with pairs of arguments -- assuming outputs is a dict of dicts)
    '''
    input_pos = outputs['in_pos'][:num_to_save]
    # all_but_outputs = inputs['positions'][:num_to_save, : - num_points * 3 * t_out]
    output_pos = outputs['tv'][:num_to_save]
    pred = outputs['pred'][:num_to_save]
    skip = outputs['skip'][:num_to_save]
    retval = {'inpos' : input_pos, 'outpos' : output_pos, 'pred' : pred, 'skip' : skip}
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


def get_loss_for_val(inputs, outputs, **loss_params):
   	return {'loss' : modelsource.l2_diff_loss_fn_w_skip(outputs, None)}



params = {
	'save_params' : {
	    'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'synth_pos',
        'exp_id': 'rs_lsl',
        'save_valid_freq': 2000,
        'save_filters_freq': 1000000,
        'cache_filters_freq': 50000,
        'save_initial_filters' : False,
        'save_to_gfs': ['inpos', 'outpos', 'pred', 'skip'],
        'cache_dir' : '/media/data/nhaber',
	},

	'model_params' : {
		'func' : modelsource.lin_square_lin,
        'cfg' : cfg_lin
        # 't_in' : T_in,
        # 't_out' : T_out,
	},

	'train_params': {
        'data_params': {
            'func': RandomParabolaRandomFutureTimeGenerator,
            't_in' : T_in,
            't_out' : T_out,
            'n_threads' : 1,
            'batch_size' : DATA_BATCH_SIZE,
            'time_bounds' : 10.,
            'accel' : [0., -1., 0.]
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
        'targets': ['in'],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.l2_diff_loss_fn_w_skip,
		# 'loss_func_kwargs' : {
		# 	'num_channels' : 3,
  #           'threshold' : DISCRETE_THRESHOLD
		# },
		'loss_func_kwargs' : {'t_out' : T_out},
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
                'func': RandomParabolaRandomFutureTimeGenerator,
                't_in' : T_in,
                't_out' : T_out,
                'n_threads' : 1,
                'batch_size' : DATA_BATCH_SIZE,
                'time_bounds' : 10.,
                'accel' : [0., -1., 0.]
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





