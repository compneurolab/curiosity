'''
A place for generating all our configurations programatically so I can write shorter cfg scripts.
Good to keep backward compatibility, duh.
'''


import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
from tfutils import base, optimizer
import os
import numpy as np
from curiosity.interaction import train, environment, data
import tensorflow as tf
import cPickle

try:
    USER = os.environ['CURIOSITY_USER']
except KeyError:
    USER = 'nick'
if USER == 'nick':
    RENDER1_HOST_ADDRESS = '10.102.2.161'
else:
    RENDER1_HOST_ADDRESS = '10.102.2.155'
NUM_BATCHES_PER_EPOCH = 1e8
NODE_5_PORT = 15871
NODE_3_PORT = 15841
DAMIAN_PORT = 24444

def generate_latent_standards(model_cfg, learning_rate = 1e-5, optimizer_class = tf.train.AdamOptimizer):
	'''
	Some stuff that I imagine will only change in minor ways across latent scripts.
	Surely speaking too soon.
	'''
	params = {'allow_growth' : True}
	params['optimizer_params'] = {
		'world_model' : {
			'act_model' : {
				'func': optimizer.ClipOptimizer,
				'optimizer_class': optimizer_class,
				'clip': True,
			},
			'fut_model' : {
                                'func': optimizer.ClipOptimizer,
                                'optimizer_class': optimizer_class,
                                'clip': True,
                }
		},
		'uncertainty_model' : {
			'func': optimizer.ClipOptimizer,
			'optimizer_class': optimizer_class,
			'clip': True,
		}

	}
	params['learning_rate_params'] = {
		'world_model' : {
			'act_model' : {
			'func': tf.train.exponential_decay,
			'learning_rate': learning_rate,
			'decay_rate': 1.,
			'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
			'staircase': True
			},
			'fut_model' : {
                        'func': tf.train.exponential_decay,
                        'learning_rate': learning_rate,
                        'decay_rate': 1.,
                        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                        'staircase': True
                }
		},
		'uncertainty_model' : {
			'func': tf.train.exponential_decay,
			'learning_rate': learning_rate,
			'decay_rate': 1.,
			'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
			'staircase': True
		}
	}
	params['train_params'] = {
		'updater_func' : train.get_latent_updater
	}
	params['model_params'] = {
		'func' : train.get_latent_models,
		'cfg' : model_cfg,
		'action_model_desc' : 'uncertainty_model'
	}
	return params


def generate_experience_replay_data_provider(force_scaling = 80., room_dims = (5., 5.), batch_size = 32, state_time_length = 2,
	image_scale = (64,64), scene_info = environment.example_scene_info, scene_len = 1024 * 32,
	history_len = 1000, do_torque = True, get_object_there_binary = False, render_host_ip=RENDER1_HOST_ADDRESS):
	my_rng = np.random.RandomState(0)
	act_dim = 8 if do_torque else 5
	return {
		'func' : train.get_batching_data_provider,
                'action_limits' : np.array([1., 1.] + [force_scaling for _ in range(act_dim - 2)]),
                'environment_params' : {
                        'random_seed' : 1,
                        'unity_seed' : 1,
                        'room_dims' : room_dims,
                        'state_memory_len' : {
                                        'depths1' : history_len + state_time_length + 1
                                },
                        'action_memory_len' : history_len + state_time_length,
                        'message_memory_len' : history_len + state_time_length,
			'other_data_memory_length' : 32,
                        'rescale_dict' : {
                                        'depths1' : image_scale
                                },
                        'USE_TDW' : True,
                        'host_address' : render_host_ip
                },

                'provider_params' : {
                        'batching_fn' : lambda hist : data.uniform_experience_replay(hist, history_len, my_rng = my_rng, batch_size = batch_size,
					get_object_there_binary = get_object_there_binary),
                        'capacity' : 5,
                        'gather_per_batch' : batch_size / 4,
                        'gather_at_beginning' : history_len + state_time_length + 1
                },

                'scene_list' : [scene_info],
                'scene_lengths' : [scene_len],
		'do_torque' : do_torque




	}


def generate_object_there_experience_replay_provider(force_scaling = 80., room_dims = (5., 5.), batch_size = 32, state_time_length = 2,
        image_scale = (64,64), scene_info = environment.example_scene_info, scene_len = 1024 * 32,
        history_len = 1000, do_torque = True, ratio = 1. / .17, num_gathered_per_batch = None, get_object_there_binary = False, render_host_ip=RENDER1_HOST_ADDRESS):
        my_rng = np.random.RandomState(0)
	act_dim = 8 if do_torque else 5
	if num_gathered_per_batch is None:
		num_gathered_per_batch = batch_size / 4
        return {
                'func' : train.get_batching_data_provider,
                'action_limits' : np.array([1., 1.] + [force_scaling for _ in range(act_dim - 2)]),
                'environment_params' : {
                        'random_seed' : 1,
                        'unity_seed' : 1,
                        'room_dims' : room_dims,
                        'state_memory_len' : {
                                        'depths1' : history_len + state_time_length + 1
                                },
                        'action_memory_len' : history_len + state_time_length,
                        'message_memory_len' : history_len,
                        'other_data_memory_length' : 32,
                        'rescale_dict' : {
                                        'depths1' : image_scale
                                },
                        'USE_TDW' : True,
                        'host_address' : render_host_ip
                },

                'provider_params' : {
                        'batching_fn' : lambda hist : data.obj_there_experience_replay(hist, history_len, my_rng = my_rng, batch_size = batch_size, there_not_there_ratio = ratio, get_object_there_binary = get_object_there_binary),
                        'capacity' : 5,
                        'gather_per_batch' : num_gathered_per_batch,
                        'gather_at_beginning' : history_len + state_time_length + 1
                },

                'scene_list' : [scene_info],
                'scene_lengths' : [scene_len],
                'do_torque' : do_torque
	}



def generate_batching_data_provider(force_scaling = 80., 
					room_dims = (5., 5.), 
					batch_size = 32, 
					state_time_length = 2, 
					image_scale = (64, 64), 
					scene_info = environment.example_scene_info, 
					scene_len = 1024 * 32, 
					do_torque = True,
                                        render_host_ip=RENDER1_HOST_ADDRESS):
	act_dim = 8 if do_torque else 5
	return {
		'func' : train.get_batching_data_provider,
		'action_limits' : np.array([1., 1.] + [force_scaling for _ in range(act_dim - 2)]),
		'environment_params' : {
			'random_seed' : 1,
			'unity_seed' : 1,
			'room_dims' : room_dims,
			'state_memory_len' : {
					'depths1' : batch_size + state_time_length
				},
			'action_memory_len' : batch_size + state_time_length - 1,
			'message_memory_len' : batch_size,
			'rescale_dict' : {
					'depths1' : image_scale
				},
			'USE_TDW' : True,
			'host_address' : render_host_ip,
		},
		
		'provider_params' : { 
			'batching_fn' : lambda hist : data.batch_FIFO(hist, batch_size = batch_size),
			'capacity' : 5,
			'gather_per_batch' : batch_size,
			'gather_at_beginning' : batch_size
		},

		'scene_list' : [scene_info],
		'scene_lengths' : [scene_len],
		'do_torque' : do_torque
	}



def query_gen_latent_save_params(location = 'freud', prefix = None, state_desc = 'depths1', load_and_save_elsewhere = False, load_and_save_same = False, portnum = 15841):
        if location == 'freud':
                CACHE_ID_PREFIX = '/media/data4/nhaber/cache'
        elif location == 'cluster':
                CACHE_ID_PREFIX = '/mnt/fs0/nhaber/cache'
        elif location == 'damian':
                CACHE_ID_PREFIX = '/data/mrowca/cache'
        else:
                raise Exception('Where are we, again?')
	exps_there_fn = os.path.join(CACHE_ID_PREFIX, 'exps_used.pkl')
	if not os.path.isfile(exps_there_fn):
		expids_used = []
	else:
		with open(exps_there_fn) as stream:
			expids_used = cPickle.load(stream)
	exp_id = None
	while exp_id is None:
		proposed = raw_input('Please enter expid: ')
		proposed = proposed if prefix is None else prefix + '_' + proposed
		if proposed not in expids_used or (load_and_save_same and proposed in expids_used):
			expids_used.append(proposed)
			exp_id = proposed
	if load_and_save_elsewhere:
		exp_id_load = raw_input('Please enter expid to load: ')
		if prefix is not None:
			exp_id_load = prefix + '_' + exp_id_load
	else:
		exp_id_load = exp_id
	with open(exps_there_fn, 'w') as stream:
		cPickle.dump(expids_used, stream)
        CACHE_DIR = os.path.join(CACHE_ID_PREFIX, exp_id)
	CACHE_DIR_LOAD = os.path.join(CACHE_ID_PREFIX, exp_id_load)
        if not os.path.exists(CACHE_DIR):
                os.mkdir(CACHE_DIR)
	if not os.path.exists(CACHE_DIR_LOAD):
		os.mkdir(CACHE_DIR_LOAD)
	params = {'save_params' : {     
                'host' : 'localhost',
                'port' : portnum,
                'dbname' : 'uncertain_agent',
                'collname' : 'uniform_action_latent',
                'exp_id' : exp_id,
                'save_valid_freq' : 1000,
        'save_filters_freq': 60000,
        'cache_filters_freq': 20000,
        'save_metrics_freq' : 1000,
        'save_initial_filters' : False,
        'cache_dir' : CACHE_DIR,
        'save_to_gfs' : ['act_pred', 'fut_pred', 'batch', 'msg', 'recent', 'loss_per_example', 'estimated_world_loss']
        }}

        params['load_params'] = {
                'exp_id' : exp_id_load,
                'load_param_dict' : None,
		'cache_dir' : CACHE_DIR_LOAD
        }
        params['what_to_save_params'] = {
                'big_save_keys' : ['fut_loss', 'act_loss', 'um_loss', 'act_pred', 'fut_pred', 'loss_per_example', 'estimated_world_loss'],
                'little_save_keys' : ['fut_loss', 'act_loss', 'um_loss'],
                'big_save_len' : 2,
                'big_save_freq' : 1000,
                'state_descriptor' : state_desc
        }
        return params

	



def generate_latent_save_params(exp_id, location = 'freud', state_desc = 'depths1'): 
	if location == 'freud':
		CACHE_ID_PREFIX = '/media/data4/nhaber/cache'
	elif location == 'cluster':
		CACHE_ID_PREFIX = '/mnt/fs0/nhaber/cache'
	else:
		raise Exception('Where are we, again?')
	CACHE_DIR = os.path.join(CACHE_ID_PREFIX, exp_id)
	if not os.path.exists(CACHE_DIR):
		os.mkdir(CACHE_DIR)
	params = {'save_params' : {	
		'host' : 'localhost',
		'port' : 15841,
		'dbname' : 'uncertain_agent',
		'collname' : 'uniform_action_latent',
		'exp_id' : exp_id,
		'save_valid_freq' : 1000,
        'save_filters_freq': 60000,
        'cache_filters_freq': 20000,
	'save_metrics_freq' : 1000,
        'save_initial_filters' : False,
	'cache_dir' : CACHE_DIR,
        'save_to_gfs' : ['act_pred', 'fut_pred', 'batch', 'msg', 'recent']
	}}
	
	params['load_params'] = {
		'exp_id' : exp_id,
		'load_param_dict' : None
	}
	params['what_to_save_params'] = {
	        'big_save_keys' : ['fut_loss', 'act_loss', 'um_loss', 'act_pred', 'fut_pred'],
	        'little_save_keys' : ['fut_loss', 'act_loss', 'um_loss'],
		'big_save_len' : 2,
		'big_save_freq' : 1000,
		'state_descriptor' : state_desc
	}
	return params



def generate_uncertainty_model_cfg(state_time_length = 2, image_shape = (64, 64), action_dim = 8, n_action_samples = 1000, loss_factor = 1., state_desc = 'depths1', heat = 1.):
	return {
					'state_shape' : [state_time_length] + list(image_shape) + [3],
					'action_dim' : action_dim,
					'n_action_samples' : n_action_samples,
					'encode' : {
						'encode_depth' : 5,
						'encode' : {
							1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 20}},
							2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 20}},
							3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 20}},
							4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}},
							5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 5}},
						}
					},
					'mlp' : {
						'hidden_depth' : 2,
						'hidden' : {1 : {'num_features' : 20, 'dropout' : .75},
									2 : {'num_features' : 1, 'activation' : 'identity'}
						}		
					},
					'state_descriptor' : state_desc,
					'loss_factor' : loss_factor,
					'heat' : heat
				}


def generate_conv_architecture_cfg(desc = 'encode', sizes = [3, 3, 3, 3, 3], strides = [2, 1, 2, 1, 2], num_filters = [20, 20, 20, 10, 4], bypass = [None, None, None, None, None], nonlinearity = None):
	retval = {}
	if nonlinearity is None:
		nonlinearity = ['relu' for _ in sizes]
	assert len(sizes) ==  len(strides) and len(num_filters) == len(strides) and len(bypass) == len(strides)
	retval[desc + '_depth'] = len(sizes)
	retval[desc] = {}
	for i, (sz, stride, nf, byp, nl) in enumerate(zip(sizes, strides, num_filters, bypass, nonlinearity)):
		retval[desc][i + 1] = {'conv' : {'filter_size' : sz, 'stride' : stride, 'num_filters' : nf}, 'bypass' : byp, 'nonlinearity' : nl}
	return retval

def generate_mlp_architecture_cfg(num_features = [20, 1], dropout = [None, None], nonlinearities = ['relu', 'identity']):
	retval = {}
	assert len(num_features) == len(dropout) and len(dropout) == len(nonlinearities)
	retval['hidden_depth'] = len(num_features)
	retval['hidden'] = {}
	for i, (nf, drop, nl) in enumerate(zip(num_features, dropout, nonlinearities)):
		retval['hidden'][i + 1] = {'num_features' : nf, 'dropout' : drop, 'activation' : nl}
	return retval

def generate_latent_marioish_world_model_cfg(state_time_length = 2, image_shape = (64, 64), action_dim = 8, act_loss_factor = 1., fut_loss_factor = 1.,
				encode_deets = {'sizes' : [3, 3, 3, 3], 'strides' : [2, 2, 2, 2], 'nf' : [32, 32, 32, 32]},
				action_deets = {'nf' : [256]},
				future_deets = {'nf' : [512]},
				act_loss_type = 'both_l2', num_classes = 21, include_previous_action = False
				):
	params = {'state_shape' : [state_time_length] + list(image_shape) + [3], 'action_shape' : [state_time_length, action_dim]}
	encode_params = {'encode' : {}, 'encode_depth' : 0}
	assert len(set([len(deet) for deet in encode_deets.values()])) == 1
	for i, (sz, stride, num_feat) in enumerate(zip(encode_deets['sizes'], encode_deets['strides'], encode_deets['nf'])):		
		encode_params['encode_depth'] += 1
		encode_params['encode'][i + 1] = {'conv' : {'filter_size' : sz, 'stride' : stride, 'num_filters' : num_feat}}
	params['encode'] = encode_params
	#action cfg construction
	act_hidden_depth = len(action_deets['nf']) + 1
	act_pred_time_length = 2 if act_loss_type == 'both_l2' else 1
	if act_loss_type != 'one_cat':
		num_classes = 1
	action_mlp_params = {'hidden_depth' : act_hidden_depth, 'hidden' : {act_hidden_depth : {'num_features' : act_pred_time_length * action_dim * num_classes, 'activation' : 'identity'}}}
	for i, nf in enumerate(action_deets['nf']):
		action_mlp_params['hidden'][i + 1] = {'num_features' : nf}
	params['action_model'] = {'mlp' : action_mlp_params, 'loss_factor' : act_loss_factor, 'loss_type' : act_loss_type, 'num_classes' : num_classes, 'include_previous_action' : include_previous_action}
	future_hidden_depth = len(future_deets['nf']) + 1
	latent_space_dim = 1
	for i in range(2):
		sz = image_shape[i]
		for stride in encode_deets['strides']:
			sz = np.ceil(float(sz) / float(stride))
		latent_space_dim *= sz
	latent_space_dim = int(latent_space_dim)
	latent_space_dim *= encode_deets['nf'][-1]
	future_mlp_params = {'hidden_depth' : future_hidden_depth, 'hidden' : {future_hidden_depth : {'num_features' : latent_space_dim, 'activation' : 'identity'}}}
	for i, nf in enumerate(future_deets['nf']):
		future_mlp_params['hidden'][i + 1] = {'num_features' : nf}
	params['future_model'] = {'mlp' : future_mlp_params, 'loss_factor' : fut_loss_factor}
	return params

def generate_latent_model_cfg(world_cfg, uncertainty_cfg, seed = 0):
	return {'world_model' : world_cfg, 'uncertainty_model' : uncertainty_cfg, 'seed' : seed}

















 
				



















