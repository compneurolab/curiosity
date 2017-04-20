import numpy as np
import os
import tensorflow as tf
import sys
import json
import copy
from tqdm import trange

from tfutils import base, data, model, optimizer, utils
from curiosity.data.threeworld_data import ThreeWorldDataProvider
import curiosity.models.vpn as modelsource
from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)

conf = 'cluster'

if conf is 'cluster':
    BASE_DIR = '/mnt/fs0/datasets/two_world_dataset'
    CACHE_DIR = '/mnt/fs0/mrowca/tfutils'
    HOST = 'localhost'
else:
    BASE_DIR = '/media/data2/new_dataset/'
    CACHE_DIR = '/media/data/mrowca/tfutils'
    HOST = 'localhost'

DATA_PATH = os.path.join(BASE_DIR, 'new_tfdata')
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, 'new_tfvaldata')
NORM_PATH = os.path.join(BASE_DIR, 'stats.pkl')

INPUT_BATCH_SIZE = 256
N_GPUS = 1
OUTPUT_BATCH_SIZE = 8 * N_GPUS
N = 2048000
NUM_BATCHES_PER_EPOCH = N // OUTPUT_BATCH_SIZE
IMAGE_SIZE_CROP = 256
TIME_DIFFERENCE = 1
SEQUENCE_LENGTH = 12
GAUSSIAN = None #['actions', 'poses']
RESIZE = {'images': [28, 64]}
RANDOM_SKIP = None
USE_VALIDATION = True
DO_TRAIN = True

seed = 0
exp_id = 'test18'

rng = np.random.RandomState(seed=seed)

def get_debug_info(inputs, outputs, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) 
    and outputs field (with pairs of arguments -- assuming outputs 
    is a dict of dicts)
    '''
    images = inputs['images'][:num_to_save]
    images = tf.cast(images, tf.uint8)

    if DO_TRAIN:
        preds = outputs['rgb'][:num_to_save]
        preds = tf.stack(preds)
        preds = tf.nn.softmax(preds)
        # maximum dimension that tf.argmax can handle is 5, so unstack here
        shape = preds.get_shape().as_list()
        preds = tf.reshape(preds, [shape[0]*shape[1]] + shape[2:])
        preds = tf.unstack(preds)
        for i, pred in enumerate(preds):
            preds[i] = tf.argmax(pred, axis=tf.rank(pred) - 1)
        preds = tf.stack(preds)
        preds = tf.cast(preds, tf.uint8)
        #actions = outputs['actions'][:num_to_save]
    else:
        preds = outputs['predicted'][:num_to_save]
        preds = tf.stack(preds)
        shape = preds.get_shape().as_list()
        preds = tf.reshape(preds, [shape[0]*shape[1]] + shape[2:])
        preds = tf.image.convert_image_dtype(preds, dtype=tf.uint8)
    retval = {'img': images, 'pred': preds,
            'decode': outputs['decode'], 'encode': outputs['encode'],
            'run_lstm': outputs['run_lstm'], 
            'ph_enc_inp': outputs['ph_enc_inp'],
            'ph_lstm_inp': outputs['ph_lstm_inp'],
            'ph_dec_inp': outputs['ph_dec_inp'],
            'ph_dec_cond': outputs['ph_dec_cond']}
    return retval

def keep_all(step_results):
    return step_results[0]

params = {
    'dont_run': True,
    'save_params' : {
        'host': HOST,
        'port': 27017,
        'dbname': 'tests',
        'collname': 'new_data',
        'exp_id': exp_id,
        'save_valid_freq': 500,
        'save_filters_freq': 50000,
        'cache_filters_freq': 2000,
        'save_metrics_freq': 50,
        'save_initial_filters' : False,
        'save_to_gfs': ['pred', 'img'],
        'cache_dir': CACHE_DIR,
    },

    'load_params': {
        'host': HOST,
        # 'port': 31001,
        # 'dbname': 'alexnet-test'
        # 'collname': 'alexnet',
        # 'exp_id': 'trainval0',
        'port': 27017,
        'dbname': 'tests',
        'collname': 'new_data',
        #'exp_id': 'trainval0',
        'exp_id': exp_id,
        #'exp_id': 'trainval2', # using screen?
        'do_restore': True,
        'load_query': None
    },

    'model_params' : {
	'func' : modelsource.parallel_model,
        'batch_size': OUTPUT_BATCH_SIZE,
        'gaussian': GAUSSIAN,
        'stats_file': NORM_PATH,
        'encoder_depth': 2,
        'decoder_depth': 4,
        'n_gpus': N_GPUS,
        'my_train': False,
        #'normalization_method': {'images': 'standard', 'actions': 'minmax'},
    }
}

if USE_VALIDATION:
    params['validation_params'] = {
        'valid0': {
            'data_params': {
                'func': ThreeWorldDataProvider,
                #'file_pattern': 'TABLE_CONTROLLED:DROP:FAST_PUSH:*.tfrecords',
                'data_path': DATA_PATH,
                'sources': ['images'],
                'n_threads': 4,
                'batch_size': INPUT_BATCH_SIZE,
                'delta_time': TIME_DIFFERENCE,
                'sequence_len': SEQUENCE_LENGTH,
                'output_format': 'sequence',
                'filters': ['is_not_teleporting'],
                'gaussian': GAUSSIAN,
                'max_random_skip': RANDOM_SKIP,
                'resize': RESIZE,
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': OUTPUT_BATCH_SIZE,
                'seed': seed,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            },
            'targets': {
                'func': get_debug_info,
                'targets' : [],
                'num_to_save' : 5
            },
        'agg_func' : keep_all,
        #'agg_func': utils.mean_dict,
        'num_steps': 10 # N_VAL // BATCH_SIZE + 1,
        #'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        #'online_agg_func': online_agg
        }
    }

if __name__ == '__main__':
    # get session and data
    base.get_params()
    sess, queues, dbinterface, valid_targets_dict = base.test_from_params(**params)
    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    # get handles to network parts
    # ops
    get_images = valid_targets_dict['valid0']['targets']['img']
    encode = valid_targets_dict['valid0']['targets']['encode']
    run_lstm = valid_targets_dict['valid0']['targets']['run_lstm']
    decode = valid_targets_dict['valid0']['targets']['decode']
    # placeholders
    ph_enc_inp = valid_targets_dict['valid0']['targets']['ph_enc_inp']
    ph_lstm_inp = valid_targets_dict['valid0']['targets']['ph_lstm_inp']
    ph_dec_inp = valid_targets_dict['valid0']['targets']['ph_dec_inp']
    ph_dec_cond = valid_targets_dict['valid0']['targets']['ph_dec_cond']
    for ex in range(1): #TODO xrange(valid_targets_dict['valid0']['num_steps']):
        # get images
        images = sess.run(get_images)[0].astype(np.float32) / 255.0
        context_images = np.zeros(list(images.shape[:-1]) + list([256]))
        for i in range(2):
            context_image = np.expand_dims(images[:,i,:,:,:], 1)
            context_images[:,i,:,:,:] = np.squeeze(sess.run(encode, 
                    feed_dict={ph_enc_inp: context_image})[0])
        encoded_images = sess.run(run_lstm, 
                feed_dict={ph_lstm_inp: context_images})[0]
        image = np.zeros(images[:,2,:,:,:].shape)
        image = np.expand_dims(image, 1)
        context = encoded_images[:,1,:,:,:]
        context = np.expand_dims(context, 1)
        print('Unrolling pixel by pixel:')
        for i in trange(images.shape[-3], desc='height'):
            for j in trange(images.shape[-2], desc='width'):
                for k in xrange(images.shape[-1]):
                    image[:,0,i,j,k] = sess.run(decode, 
                            feed_dict={ph_dec_inp: image,
                                ph_dec_cond: context})[0][:,0,i,j,k]
