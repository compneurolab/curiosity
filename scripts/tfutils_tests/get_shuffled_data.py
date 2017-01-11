import numpy as np
import os
import sys
import tensorflow as tf
from tfutils import base, data, model, optimizer
from curiosity.data.images_futures_and_actions import FuturePredictionData 


DATA_PATH = '/media/data2/one_world_dataset/dataset.hdf5'
BATCH_SIZE = 256
N = 2048000
NUM_BATCHES_PER_EPOCH = N // BATCH_SIZE
IMAGE_SIZE_CROP = 256

def shuffle_net(inputs, train=False, **kwargs):
    m = model.ConvNet(**kwargs)
	#TODO FILL HERE!
    return m

def simple_return(inputs, outputs, target):
    return outputs

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

params = {
    'save_params': {
        'host': 'localhost',
        'port': 29101,
        'dbname': 'randomshuffle-test',
        'collname': 'randomshuffle',
        'exp_id': 'test1',

        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 5,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 3000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 3000,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        # 'host': 'localhost',
        # 'port': 31001,
        # 'dbname': 'alexnet-test',
        # 'collname': 'alexnet',
        # 'exp_id': 'trainval0',
        'do_restore': False,
        'load_query': None
    },

    'model_params': {
        'func': shuffle_net,
        'seed': 0,
        'norm': False  # do you want local response normalization?
    },

    'train_params': {
        'data_params': {
            'func': FuturePredictionData,
            'data_path': DATA_PATH,
            'crop_size': IMAGE_SIZE_CROP,
	    'random_time': False,
            'batch_size': 1
        },
        'queue_params': {
            'queue_type': 'fifo', #TODO switch to random
            'batch_size': BATCH_SIZE,
            'n_threads': 4,
            'seed': 0,
        },
        'thres_loss': 1000,
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
    },

    'loss_params': {
        'targets': 'future_actions',
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': .01,
        'decay_rate': .95,
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
        'topn': {
            'data_params': {
                'func': FuturePredictionData,
                'data_path': DATA_PATH,  # path to image database
                'random_time': False,
                'crop_size': IMAGE_SIZE_CROP,  # size after cropping an image
            },
            'targets': {
                'func': simple_return,
                'target': 'future_actions',
            },
            'queue_params': {
                'queue_type': 'fifo', #TODO switch to random
                'batch_size': BATCH_SIZE,
                'n_threads': 4,
                'seed': 0,
            },
            'num_steps': 1 # N_VAL // BATCH_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        },
    },

    'log_device_placement': False,  # if variable placement has to be logged
}

if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
