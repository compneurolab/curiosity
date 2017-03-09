'''
Training a squeezed-down version of explicit positions. More significant filter, just one object (hopefully the one that is moving).
'''

uniform_width = 50

cfg_0 = {
	'hidden_depth' : 2,
	'hidden' : {
		1 : {'num_features' : uniform_width},
        2 : {'num_features' : uniform_width},
	}
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


cfg_one_square = {
    'first_lin' : 30
}

cfg_linear = {
    'hidden_depth' : 0
}

import numpy as np
import os
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import json

from tfutils import base, data, model, optimizer, utils
from curiosity.data.explicit_positions import SqueezedPositionPrediction 
import curiosity.models.explicit_position_models as modelsource

DATA_PATH = '/media/data2/one_world_dataset/tfdata'
VALIDATION_DATA_PATH = '/media/data2/one_world_dataset/tfvaldata'
STATS_FILE = '/media/data/one_world_dataset/dataset_stats.pkl'
DATA_BATCH_SIZE = 256
MODEL_BATCH_SIZE = 256
N = 2048000
NUM_BATCHES_PER_EPOCH = N // MODEL_BATCH_SIZE
seed = 0
T_in = 3
T_out = 1
MAX_SKIP = 4
SEQ_LEN = T_in + T_out + MAX_SKIP

#TODO should keep loss for all!
def append_every_kth(x, y, step, k):
    if x is None:
        x = []
    if step % k == 0:
        x.append(y)
    return x



def get_ins_and_outs(inputs, outputs, num_to_save = 1, seq_len = 6, t_out = 3, t_in = 3, num_points = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) and outputs field (with pairs of arguments -- assuming outputs is a dict of dicts)
    '''
    input_pos = outputs['in_pos'][:num_to_save]
    actions = inputs['corresponding_actions'][:num_to_save]
    output_pos = outputs['tv'][:num_to_save]
    all_pos = inputs['pos_squeezed'][:num_to_save]
    pred = outputs['pred'][:num_to_save]
    skip = outputs['skip'][:num_to_save]
    retval = {'inpos' : input_pos, 'outpos' : output_pos, 'pred' : pred, 'act' : actions, 'skip' : skip, 'pos' : all_pos}
    retval.update(get_loss_for_val(inputs, outputs, t_out = t_out, t_in = t_in, num_points = num_points, ** loss_params))
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
    return {'loss' : modelsource.l2_diff_loss_fn_w_skip(outputs, inputs['pos_in'], t_in = loss_params['t_in'], t_out = loss_params['t_out'], num_points = loss_params['num_points'])}



params = {
	'save_params' : {
	    'host': 'localhost',
        'port': 27017,
        'dbname': 'future_pred_test',
        'collname': 'positions',
        'exp_id': 'squ_rand18',
        'save_valid_freq': 2000,
        'save_filters_freq': 1000000,
        'cache_filters_freq': 50000,
        'save_initial_filters' : False,
        'save_to_gfs': ['inpos', 'outpos', 'pred', 'act', 'skip', 'pos'],
        'cache_dir' : '/media/data/nhaber',
	},

	'model_params' : {
		'func' : modelsource.variable_skip_mlp,
		'cfg' : cfg_0,
        'stddev' : .01,
        'activation' : 'relu'
	},

	'train_params': {
        'data_params': {
            'func': SqueezedPositionPrediction,
            'seq_len' : SEQ_LEN,
            'data_path': DATA_PATH,
    	    'batch_size': DATA_BATCH_SIZE,
            'n_threads' : 4,
            't_in' : T_in,
            't_out' : T_out,
            'shuffle' : True,
            'shuffle_seed' : 1234,
            'filters' : ['mask_squeezed'],
            'normalize_actions' : 'custom',
            'stats_file' : STATS_FILE,
            'manual_coord_scaling' : [1., 1., 1.]
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': MODEL_BATCH_SIZE,
            'seed': 0,
    	    'capacity': MODEL_BATCH_SIZE * 14,
        },
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'thres_loss' : float('inf'),
    },


    'loss_params': {
        'targets': ['pos_in'],
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': modelsource.l2_diff_loss_fn_w_skip,
		# 'loss_func_kwargs' : {
		# 	'num_channels' : 3,
  #           'threshold' : DISCRETE_THRESHOLD
		# },
		'loss_func_kwargs' : {'t_in' : T_in, 't_out' : T_out, 'num_points' : 1},
		'loss_per_case_func_params' : {}
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-5,
        'decay_rate': 0.95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'validation_params': {
        'valid0': {
            'data_params': {
                'func': SqueezedPositionPrediction,
                'seq_len' : SEQ_LEN,
                'data_path': VALIDATION_DATA_PATH,
                'batch_size': DATA_BATCH_SIZE,
                'n_threads' : 1,
                't_in' : T_in,
                't_out' : T_out,
                'shuffle' : True,
                'shuffle_seed' : 1234,
                'filters' : ['mask_squeezed'],
                'normalize_actions' : 'custom',
                'stats_file' : STATS_FILE,
                'manual_coord_scaling' : [1., 1., 1.]
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': MODEL_BATCH_SIZE,
                'seed': 0,
                'capacity': MODEL_BATCH_SIZE * 2,
                # 'n_threads' : 4

            },
        'targets': {
                'func': get_ins_and_outs,
                'targets' : [],
                'num_to_save' : 1,
                't_out' : T_out,
                'seq_len' : SEQ_LEN
            },
        'agg_func' : mean_losses_keep_rest,
        'online_agg_func' : lambda x, y, step : append_every_kth(x, y, step, 10),
        'num_steps': 100
        },
        # 'valid1': {
        #     'data_params': {
        #         'func': SqueezedPositionPrediction,
        #         'seq_len' : SEQ_LEN,
        #         'data_path': DATA_PATH,
        #         'batch_size': DATA_BATCH_SIZE,
        #         'n_threads' : 1,
        #         't_in' : T_in,
        #         't_out' : T_out,
        #         'shuffle' : True,
        #         'shuffle_seed' : 1234,
        #         'filters' : ['mask_squeezed'],
        #         'normalize_actions' : 'custom',
        #         'stats_file' : STATS_FILE,
        #         'manual_coord_scaling' : [1., 1., 1.]
        #     },
        #     'queue_params': {
        #         'queue_type': 'random',
        #         'batch_size': MODEL_BATCH_SIZE,
        #         'seed': 0,
        #       'capacity': MODEL_BATCH_SIZE * 10,
        #         # 'n_threads' : 4

        #     },
        # 'targets': {
        #         'func': get_ins_and_outs,
        #         'targets' : [],
        #         'num_to_save' : 1,
        #         't_out' : T_out,
        #         'seq_len' : SEQ_LEN
        #     },
        # 'agg_func' : mean_losses_keep_rest,
        # 'online_agg_func' : lambda x, y, step : append_every_kth(x, y, step, 10),
        # 'num_steps': 100
        # }
    }



}


if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)





