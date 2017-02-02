import numpy as np
import os
import sys
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
from curiosity.data.images_futures_and_actions import FuturePredictionData 
from tfutils.data import TFRecordsDataProvider

#DATA_PATH = '/media/data/one_world_dataset/dataset.hdf5'
DATA_PATH = '/media/data2/one_world_dataset/dataset.tfrecords'
DATA_PATH = '/home/mrowca/data/dataset_images_parsed_actions8.tfrecords'
#DATA_PATH = '/media/data2/one_world_dataset/dataset.lmdb'
#DATA_PATH = '/media/data2/one_world_dataset/data_format_tests/data_tf.lmdb'
BATCH_SIZE = 256
N = 128000 #2048000
NUM_BATCHES_PER_EPOCH = N // BATCH_SIZE
IMAGE_SIZE_CROP = 256

def shuffle_net(inputs, train=False, **kwargs):
    print(inputs['images'])
    inp = tf.cast(inputs['images'], tf.float32)
    inp = tf.divide(inp, 255)
    act = inputs['parsed_actions'] # THIS IS NEW
    #inp = inputs['images']
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
    return {'inputs': inputs['images'], }#'outputs': inputs['future_ids']}

def dummy_loss(**kwargs):
    kwargs['labels'] = tf.cast(kwargs['labels'], tf.int32)[:,0]
    return tf.nn.sparse_softmax_cross_entropy_with_logits(**kwargs)
    

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
        'exp_id': 'test12',
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
            #'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],
	    #'random_time': False,
            'min_time_difference': 5,
	    'batch_size': 256,
            #'tfsource': DATA_PATH,
            'n_threads': 4,
            'output_format': {'images': 'pairs', 'actions': 'sequence'},
            'use_object_ids': True,
            #'sourcedict': {'images': tf.string, 'parsed_actions': tf.string},
            #'imagelist': ['images'],
        },
        'queue_params': {
            'queue_type': 'fifo',
            'batch_size': BATCH_SIZE,
            'seed': 0,
	    'capacity': BATCH_SIZE * 100
        },
        'num_steps': 10, #90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
    },

    'loss_params': {
        'targets': 'parsed_actions',
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': dummy_loss,
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
                #'random_time': False,
                #'crop_size': [IMAGE_SIZE_CROP, IMAGE_SIZE_CROP],  # size after cropping an image
		'min_time_difference': 5,
                'output_format': {'images': 'pairs', 'actions': 'sequence'},  
                'use_object_ids': True,
		'batch_size': 256,
                'n_threads': 4,
                #'batch_size': 256,
                #'tfsource': DATA_PATH,
                #'sourcedict': {'images': tf.string, 'parsed_actions': tf.string},
                #'imagelist': ['images'],
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE,
                'seed': 0,
		'capacity': BATCH_SIZE * 100,
            },
	    'targets': {
                'func': simple_return,
                'target': 'parsed_actions',
            },
	    'agg_func': utils.mean_dict,
            'num_steps': 1, # N_VAL // BATCH_SIZE + 1,
	    #'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            #'online_agg_func': online_agg
        },
    },

    'log_device_placement': False,  # if variable placement has to be logged
}

if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
