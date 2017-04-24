import numpy as np
import os
import tensorflow as tf
import sys
import json
import copy
import cPickle
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
SEGMENTATION = ['actions', 'positions']
RESIZE = {'images': [28, 64], 'objects': [28, 64]}
RANDOM_SKIP = None
USE_VALIDATION = True

seed = 0
exp_id = 'test28'

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
            'object_data': inputs['object_data'][:num_to_save],
            'decode': outputs['decode'], 
            'encode': outputs['encode'],
            'run_lstm': outputs['run_lstm'], 
            'ph_enc_inp': outputs['ph_enc_inp'],
            'ph_enc_cond': outputs['ph_enc_cond'],
            'ph_lstm_inp': outputs['ph_lstm_inp'],
            'ph_dec_inp': outputs['ph_dec_inp'],
            'ph_dec_cond': outputs['ph_dec_cond'],}
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
                'sources': ['images', 'actions', 'objects', 'object_data'],
                'n_threads': 1,
                'batch_size': INPUT_BATCH_SIZE,
                'delta_time': TIME_DIFFERENCE,
                'sequence_len': SEQUENCE_LENGTH,
                'output_format': 'sequence',
                'filters': ['is_not_teleporting',], #'is_object_there'],
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
    get_object_data = valid_targets_dict['valid0']['targets']['object_data']
    encode = valid_targets_dict['valid0']['targets']['encode']
    run_lstm = valid_targets_dict['valid0']['targets']['run_lstm']
    decode = valid_targets_dict['valid0']['targets']['decode']
    # placeholders
    ph_enc_inp = valid_targets_dict['valid0']['targets']['ph_enc_inp']
    ph_enc_cond = valid_targets_dict['valid0']['targets']['ph_enc_cond']
    ph_lstm_inp = valid_targets_dict['valid0']['targets']['ph_lstm_inp']
    ph_dec_inp = valid_targets_dict['valid0']['targets']['ph_dec_inp']
    ph_dec_cond = valid_targets_dict['valid0']['targets']['ph_dec_cond']
    # unroll across time
    n_context = 2
    for ex in xrange(valid_targets_dict['valid0']['num_steps']):
        # get inputs: images, actions, and segmented object images
        images, actions, objects, object_data = sess.run([get_images, 
            get_actions, 
            get_objects,
            get_object_data])
        images = images[0].astype(np.float32) / 255.0
        # construct action segmentation mask
        objects = objects[0][:,:,:,:,0] * (256**2) + \
                objects[0][:,:,:,:,1] * 256 + objects[0][:,:,:,:,2]
        pos_id = object_data[0][:,:,0,0]
        action_id = actions[0][:,:,8]
        if (action_id != pos_id).all():
            print("WARNING: action_id != pos_id")
        forces = actions[0][:,:,0:6]
        objects = (objects == np.ones(objects.shape) * 
                pos_id[:,:,np.newaxis, np.newaxis]).astype(np.float32)
        acted = np.unique(np.nonzero(objects)[0])
        if len(acted) > 0:
            print('Actions present in evaluated batch for examples: ' + str(acted))
        action_masks = np.expand_dims(np.expand_dims(forces, 2), 2) \
                * np.expand_dims(objects,4)
        position_masks = np.expand_dims(objects,4)
        poses = objects.copy() * 255 #ground truth pos
        # save input data
        inputs = {'images': images, 
                'actions': actions, 
                'objects': objects, 
                'object_data': object_data,
                'action_masks': action_masks,
                'position_masks': position_masks}
        with open('inputs'+str(ex)+'.pkl', 'w') as f:
            cPickle.dump(inputs, f) 

        # encode context images
        context_images = np.zeros(list(images.shape[:-1]) + list([256]))
        #zero out later masks as they are predicted
        action_masks[:,n_context:] = 0
        position_masks[:,n_context:] = 0
        print('Encoding context:')
        for im in trange(n_context, desc='timestep'):
            context_image = np.expand_dims(images[:,im], 1)
            action_mask = np.expand_dims(action_masks[:,im], 1)
            position_mask = np.expand_dims(position_masks[:,im], 1)
            enc_cond = np.concatenate((action_mask, position_mask), axis=4)
            context_images[:,im] = np.squeeze(sess.run(encode, 
                    feed_dict={ph_enc_inp: context_image,
                        ph_enc_cond: enc_cond})[0])
        # predict images pixel by pixel, one after another
        print('Generating images pixel by pixel:')
        predicted_images = []
        predicted_poses = []
        for im in trange(n_context, images.shape[1], desc='timestep'):
            encoded_images = sess.run(run_lstm,
                    feed_dict={ph_lstm_inp: context_images})[0]
            image = np.zeros(images[:,im].shape)
            image = np.expand_dims(image, 1)
            pos = np.zeros(images[:,im,:,:,0].shape)
            context = np.expand_dims(encoded_images[:,im-1], 1)
            action_mask = np.expand_dims(action_masks[:,im-1], 1)
            position_mask = np.expand_dims(position_masks[:,im-1], 1)
            dec_cond = np.concatenate((context, action_mask, position_mask), axis=4)
            for i in trange(images.shape[-3], desc='height', leave=False):
                for j in trange(images.shape[-2], desc='width'):
                    #for k in xrange(images.shape[-1]): # predict all channels at once
                        image_pos = sess.run(decode,
                                feed_dict={ph_dec_inp: image,
                                    ph_dec_cond: dec_cond})[0]
                        image[:,0,i,j] = image_pos[:,0,i,j,0:3]
                        pos[:,i,j] = image_pos[:,0,i,j,3]
            # current action
            action_masks[:,im] = forces[:,im, np.newaxis, np.newaxis] \
                    * np.expand_dims(pos,3)
            position_masks[:,im] = np.expand_dims(pos,3)
            action_mask = np.expand_dims(action_masks[:,im], 1)
            position_mask = np.expand_dims(position_masks[:,im], 1)
            enc_cond = np.concatenate((action_mask, position_mask), axis=4)
            context_images[:,im] = np.squeeze(sess.run(encode,
                feed_dict={ph_enc_inp: image,
                    ph_enc_cond: enc_cond})[0])
            predicted_images.append(np.squeeze(image))
            predicted_poses.append(np.squeeze(pos))
        predicted_images = np.stack(predicted_images, axis=1)
        predicted_poses = np.stack(predicted_poses, axis=1)
        results = {'pred_poses': predicted_poses,
                'pred_images': predicted_images,
                'gt_poses': poses,
                'gt_images': images,
                'ex_acted': acted}
        with open('results'+str(ex)+'.pkl', 'w') as f:
            cPickle.dump(results, f)
