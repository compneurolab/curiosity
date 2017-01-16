import numpy as np
import os
import sys
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
from curiosity.data.images_futures_and_actions import FuturePredictionData 


DATA_PATH = '/media/data2/one_world_dataset/old_dataset.hdf5'
BATCH_SIZE = 256
N = 2048000
NUM_BATCHES_PER_EPOCH = N // BATCH_SIZE
IMAGE_SIZE_CROP = 256

def shuffle_net(inputs, train=False, **kwargs):
    inp = inputs['images']
    print("\033[91myaaaaay\033[0m")
    print(inp)     
    tf.Print(inp, [inp], message="This is a: ")

    m = model.ConvNet(**kwargs)

    with tf.contrib.framework.arg_scope([m.fc], init='trunc_norm', stddev=.01,
                                        bias=0, activation='relu', dropout=None):
        with tf.variable_scope('hidden1'):
            m.fc(128, in_layer=inp)

        with tf.variable_scope('hidden2'):
            m.fc(32)

        with tf.variable_scope('softmax_linear'):
            m.fc(18, activation=None)
    
    return m.output, m.params

def simple_return(inputs, outputs, target):
    tf.Print(inputs['images'], [inputs['images']], message="This is a: ")
    return {'inputs': inputs['ids'], 'outputs': inputs['future_ids']}

def dummy_loss(**kwargs):
    return 1.0

"""
def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res
"""

params = {
    'save_params': {
        'host': 'localhost',
        'port': 27017,
        'dbname': 'randomshuffle-test',
        'collname': 'randomshuffle',
        'exp_id': 'test1',
        'save_valid_freq': 3000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 3000,
    },

    'model_params': {
        'func': shuffle_net,
    },

    'train_params': {
        'data_params': {
            'func': FuturePredictionData,
            'data_path': DATA_PATH,
            'crop_size': IMAGE_SIZE_CROP,
	    'random_time': False,
            'min_time_difference': 10,
	    'batch_size': 256
        },
        'queue_params': {
            'queue_type': 'fifo', #TODO switch to random
            'batch_size': BATCH_SIZE,
            'n_threads': 1,
            'seed': 0,
        },
        'num_steps': 1 #90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
    },

    'loss_params': {
        'targets': 'future_actions',
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
    },

    'learning_rate_params': {
        'learning_rate': 0.05,
        'decay_rate': 0.95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'validation_params': {
        'valid0': {
            'data_params': {
                'func': FuturePredictionData,
                'data_path': DATA_PATH,  # path to image database
                'random_time': False,
                'crop_size': IMAGE_SIZE_CROP,  # size after cropping an image
		'min_time_difference': 10,
		'batch_size': 256,
            },
            'queue_params': {
                'queue_type': 'fifo', #TODO switch to random
                'batch_size': BATCH_SIZE,
                'n_threads': 1,
                'seed': 0,
            },
	    'targets': {
                'func': simple_return,
                'target': 'future_actions',
            },
	    'agg_func': utils.mean_dict,
            'num_steps': 1 # N_VAL // BATCH_SIZE + 1,
            #'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            #'online_agg_func': online_agg
        },
    },

    'log_device_placement': False,  # if variable placement has to be logged
}

if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
