import numpy as np
import os
import tensorflow as tf
import sys
import json
import copy
from tqdm import trange

from tfutils import base, data, model, optimizer, utils
from curiosity.data.threeworld_data import ThreeWorldDataProvider
import curiosity.models.physics_vpn as modelsource
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
SEGMENTATION = ['actions']
RESIZE = {'images': [28, 64], 'objects': [28, 64]}
RANDOM_SKIP = None
USE_VALIDATION = True

seed = 0
exp_id = 'test22'

rng = np.random.RandomState(seed=seed)

def get_debug_info(inputs, outputs, num_to_save = 1, **loss_params):
    '''
    Gives you input tensors and output tensors.

    Assumes to_extract has an inputs field (with list of arguments) 
    and outputs field (with pairs of arguments -- assuming outputs 
    is a dict of dicts)
    '''
    retval = {'images': inputs['images'][:num_to_save], 
            'actions': inputs['actions'][:num_to_save], 
            'objects': inputs['objects'][:num_to_save],
            'decode': outputs['decode'], 
            'encode': outputs['encode'],
            'run_lstm': outputs['run_lstm'], 
            'ph_enc_inp': outputs['ph_enc_inp'],
            'ph_enc_cond': outputs['ph_enc_cond'],
            'ph_lstm_inp': outputs['ph_lstm_inp'],
            'ph_dec_inp': outputs['ph_dec_inp'],
            'ph_dec_cond_past': outputs['ph_dec_cond_past'],
            'ph_dec_cond_act': outputs['ph_dec_cond_act'],}
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
        'save_to_gfs': [],
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
        'segmentation': SEGMENTATION,
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
                'sources': ['images', 'actions', 'objects'],
                'n_threads': 1,
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
        'num_steps': 1 # N_VAL // BATCH_SIZE + 1,
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
    get_images = valid_targets_dict['valid0']['targets']['images']
    get_actions = valid_targets_dict['valid0']['targets']['actions']
    get_objects = valid_targets_dict['valid0']['targets']['objects']
    encode = valid_targets_dict['valid0']['targets']['encode']
    run_lstm = valid_targets_dict['valid0']['targets']['run_lstm']
    decode = valid_targets_dict['valid0']['targets']['decode']
    # placeholders
    ph_enc_inp = valid_targets_dict['valid0']['targets']['ph_enc_inp']
    ph_enc_cond = valid_targets_dict['valid0']['targets']['ph_enc_cond']
    ph_lstm_inp = valid_targets_dict['valid0']['targets']['ph_lstm_inp']
    ph_dec_inp = valid_targets_dict['valid0']['targets']['ph_dec_inp']
    ph_dec_cond_past = valid_targets_dict['valid0']['targets']['ph_dec_cond_past']
    ph_dec_cond_act = valid_targets_dict['valid0']['targets']['ph_dec_cond_act']
    # unroll across time
    n_context = 2
    for ex in xrange(valid_targets_dict['valid0']['num_steps']):
        # get inputs: images, actions, and segmented object images
        images, actions, objects = sess.run([get_images, get_actions, get_objects])
        print(images[0].shape, actions[0].shape, objects[0].shape)
        images = images[0].astype(np.float32) / 255.0
        # construct action segmentation mask
        objects = objects[0][:,:,:,:,0] * (256**2) + \
                objects[0][:,:,:,:,1] * 256 + objects[0][:,:,:,:,2]
        forces = actions[0][:,:,0:6]
        action_id = actions[0][:,:,8]
        objects[objects == action_id] = 1
        objects[objects != action_id] = 0
        action_masks = np.expand_dims(np.expand_dims(forces, 2), 2) \
                * np.expand_dims(objects,4)
        gt_pos = objects.copy() * 255 #ground truth pos
        # encode context images
        context_images = np.zeros(list(images.shape[:-1]) + list([256]))
        action_masks[:,0:n_context] = 0 #zero out later masks as they are predicted
        print('Encoding context:')
        for im in trange(n_context, desc='timestep'):
            context_image = np.expand_dims(images[:,im], 1)
            action_mask = np.expand_dims(action_masks[:,im], 1)
            context_images[:,im] = np.squeeze(sess.run(encode, 
                    feed_dict={ph_enc_inp: context_image,
                        ph_enc_cond: action_mask})[0])
        # predict images pixel by pixel, one after another
        print('Generating images pixel by pixel:')
        predicted_images = []
        for im in trange(n_context, images.shape[1], desc='timestep'):
            encoded_images = sess.run(run_lstm,
                    feed_dict={ph_lstm_inp: context_images})[0]
            image = np.zeros(images[:,im].shape)
            image = np.expand_dims(image, 1)
            pos = np.zeros(images[:,im,:,:,0].shape)
            context = np.expand_dims(encoded_images[:,im-1], 1)
            action_mask = np.expand_dims(action_masks[:,im-1], 1)
            for i in trange(images.shape[-3], desc='height', leave=False):
                for j in trange(images.shape[-2], desc='width'):
                    #for k in xrange(images.shape[-1]): # predict all channels at once
                        image_pos = sess.run(decode,
                                feed_dict={ph_dec_inp: image,
                                    ph_dec_cond_past: context,
                                    ph_dec_cond_act: action_mask})[0]
                        image[:,0,i,j] = image_pos[:,0,i,j,0:3]
                        pos[:,i,j] = image_pos[:,0,i,j,3]
            # current action
            action_masks[:,im] = forces[:,im, np.newaxis, np.newaxis] \
                    * np.expand_dims(pos,3)
            action_mask = np.expand_dims(action_masks[:,im], 1)
            context_images[:,im] = np.squeeze(sess.run(encode,
                feed_dict={ph_enc_inp: image,
                    ph_enc_cond: action_mask})[0])
            predicted_images.append(np.squeeze(image))
        predicted_images = np.stack(predicted_images, axis=1)
        np.save('predicted_images.npy', predicted_images)
        np.save('gt_images.npy', images)
