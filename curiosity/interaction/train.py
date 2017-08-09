'''
train.py
has the actual loop for training, given all the parameters and things.
This should eventually be like tfutils.base, though for now we specialize a lot.
'''

import curiosity.interaction.environment as environment
import curiosity.interaction.data as data
from curiosity.interaction.data import SimpleSamplingInteractiveDataProvider, SillyLittleListerator
from curiosity.interaction.models import UncertaintyModel, DepthFuturePredictionWorldModel, UniformActionSampler, DamianModel, LatentSpaceWorldModel
from curiosity.interaction.update_step import UncertaintyUpdater, UncertaintyPostprocessor, DamianWMUncertaintyUpdater, LatentUncertaintyUpdater, ExperienceReplayPostprocessor, DataWriteUpdater
import tensorflow as tf
import os
import cPickle
import tfutils.base as base
import time
import cv2
import numpy as np
import tqdm


RENDER_2_ADDY = '10.102.2.162'
RENDER_1_ADDY = '10.102.2.161'

def get_default_postprocessor(what_to_save_params):
	return UncertaintyPostprocessor(** what_to_save_params)

def get_experience_replay_postprocessor(what_to_save_params):
	return ExperienceReplayPostprocessor(** what_to_save_params)

def get_models_damianworld(cfg):
	world_model = DamianModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'])
	return {'world_model' : world_model, 'uncertainty_model' : uncertainty_model}


def get_default_models(cfg):
	world_model = DepthFuturePredictionWorldModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'])
	return {'world_model' : world_model, 'uncertainty_model' : uncertainty_model}

def get_latent_models(cfg):
	world_model = LatentSpaceWorldModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'], world_model)
	return {'world_model' : world_model, 'uncertainty_model' : uncertainty_model}

def get_batching_data_provider(data_params, model_params, action_model):
	assert set(data_params.keys()) == set(['func', 'environment_params', 'scene_list', 'scene_lengths', 'provider_params', 'action_limits', 'do_torque'])
	action_to_message = lambda action, env : environment.normalized_action_to_ego_force_torque(action, env, data_params['action_limits'], wall_safety = .5, do_torque = data_params['do_torque'])
	env = environment.Environment(action_to_message_fn = action_to_message, ** data_params['environment_params'])
	scene_infos = data.SillyLittleListerator(data_params['scene_list'])
	steps_per_scene = data.SillyLittleListerator(data_params['scene_lengths'])
	data_provider = data.BSInteractiveDataProvider(env, action_model, scene_infos, steps_per_scene, UniformActionSampler(model_params['cfg']), ** data_params['provider_params'])
	return data_provider

def get_default_data_provider(data_params, model_params, action_model):
	action_to_message = lambda action, env : environment.normalized_action_to_ego_force_torque(action, env, data_params['action_limits'], wall_safety = .5)
	env = environment.Environment(action_to_message_fn = action_to_message, ** data_params['environment_params'])
	scene_infos = data.SillyLittleListerator(data_params['scene_list'])
	steps_per_scene = data.SillyLittleListerator(data_params['scene_lengths'])
	data_provider = SimpleSamplingInteractiveDataProvider(env, action_model, 1, scene_infos, steps_per_scene, UniformActionSampler(model_params['cfg']), data_params['capacity'])
	return data_provider


def get_latent_updater(models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params):
	world_model = models['world_model']
	uncertainty_model = models['uncertainty_model']
        return LatentUncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params = updater_params)


def get_default_updater(models, data_provider, optimizer_params, learning_rate_params, postprocessor):
	world_model = models['world_model']
	uncertainty_model = models['uncertainty_model']
	return UncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor)

def get_damian_updater(models, data_provider, optimizer_params, learning_rate_params, postprocessor):
	world_model = models['world_model']
        uncertainty_model = models['uncertainty_model']
        return DamianWMUncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor)


DEFAULT_WHAT_TO_SAVE_PARAMS = {
		'big_save_keys' : ['um_loss', 'wm_loss', 'wm_given', 'wm_pred', 'wm_tv'],
		'little_save_keys' : ['um_loss', 'wm_loss'],
		'big_save_len' : 100,
		'big_save_freq' : 10000,

	}

STATE_DESC = 'depths1'

LATENT_WHAT_TO_SAVE_PARAMS = {
	'big_save_keys' : ['fut_loss', 'act_loss', 'um_loss', 'encoding_i', 'encoding_f', 'act_pred', 'fut_pred'],
	'little_save_keys' : ['fut_loss', 'act_loss', 'um_loss'],
	'big_save_len' : 100,
	'big_save_freq' : 100,
	'state_descriptor' : STATE_DESC
}



def get_session(gpu_params):
	log_device_placement = gpu_params.get('log_device_placement', False)
	inter_op_parallelism_threads = gpu_params.get('inter_op_parallelism_threads', 40)
	allow_growth = gpu_params.get('allow_growth', True)
	per_process_gpu_memory_fraction = gpu_params.get('per_process_gpu_memory_fraction')
	config = tf.ConfigProto(allow_soft_placement = True,
                log_device_placement = log_device_placement, inter_op_parallelism_threads = inter_op_parallelism_threads)
        if allow_growth:
                #including this weird conditional because I'm running into a weird bug
                config.gpu_options.allow_growth = allow_growth
        if per_process_gpu_memory_fraction is not None:
                print('limiting mem fraction')
                config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        sess = tf.Session(config = config)
	return sess



def test_from_params(
		save_params,
		load_params,
		model_params,
		validate_params,
		data_params,
		gpu_params = {}
		):
	model_cfg = model_params['cfg']
	models_constructor = model_params['func']
	models = models_constructor(model_cfg)
	
	action_model = models[model_params['action_model_desc']]
	
	data_provider = data_params['func'](data_params, model_params, action_model)
	
	validater = validate_params['func'](models, data_provider, ** validate_params['kwargs'])
	num_steps = validate_params['num_steps']

	params = {'save_params' : save_params,
		'load_params' : load_params,
		'model_params' : model_params,
		'validate_params' : validate_params,
		'data_params' : data_params,
		'gpu_params' : gpu_params
		}

	sess = get_session(gpu_params)
	dbinterface = base.DBInterface(sess = sess, params = params, save_params = save_params, load_params = load_params)

        dbinterface.initialize()
        data_provider.start_runner(sess)

	test(sess, validater, dbinterface, num_steps)

def test(sess, validater, dbinterface, num_steps):
	for _step in tqdm.trange(num_steps):
		dbinterface.start_time_step = time.time()
		res = validater.run(sess)
		res = {'valid' : res}
		dbinterface.save(valid_res = res, validation_only = True)
	dbinterface.sync_with_host()



def save_data_without_training(
		model_params,
		train_params,
		data_params,
		allow_growth = True,
		inter_op_parallelism_threads = 40,
		log_device_placement = False,
		per_process_gpu_memory_fraction = None

):
	model_cfg = model_params['cfg']
	models_constructor = model_params['func']
	models = models_constructor(model_cfg)
	action_model = models[model_params['action_model_desc']]
	assert action_model.just_random == True
	
	data_provider = data_params['func'](data_params, model_params, action_model)
	updater = DataWriteUpdater(data_provider, train_params['updater_kwargs'])

	N_save = train_params['updater_kwargs']['N_save']
	batch_size = data_params['provider_params']['gather_per_batch']
	n_batches = N_save / batch_size

        config = tf.ConfigProto(allow_soft_placement = True,
                log_device_placement = log_device_placement, inter_op_parallelism_threads = inter_op_parallelism_threads)
        if allow_growth:
                #including this weird conditional because I'm running into a weird bug
                config.gpu_options.allow_growth = allow_growth
        if per_process_gpu_memory_fraction is not None:
                print('limiting mem fraction')
                config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        sess = tf.Session(config = config)

	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	data_provider.start_runner(sess)

	print('About to save ' + str(n_batches) + ' batches')

	
	for i in range(n_batches):
		print(i)
		updater.update()

	updater.close()




def train_from_params(
		save_params,
		model_params,
		train_params,
		learning_rate_params = None,
		optimizer_params = None,
		what_to_save_params = None,
		log_device_placement = False,
		load_params = None,
		data_params = None,
		inter_op_parallelism_threads = 40,
		allow_growth = False,
		per_process_gpu_memory_fraction = None,
		postprocessor_params = None
	):
	model_cfg = model_params['cfg']
	models_constructor = model_params['func']
	models = models_constructor(model_cfg)

	action_model = models[model_params['action_model_desc']]

	data_provider = data_params['func'](data_params, model_params, action_model)

	if postprocessor_params is None:
		postprocessor = get_default_postprocessor(what_to_save_params)
	else:
		postprocessor = postprocessor_params['func'](what_to_save_params)

	updater = train_params['updater_func'](models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params = train_params['updater_kwargs'])

	params = {'save_params' : save_params, 'model_params' : model_params, 'train_params' : train_params, 
		'optimizer_params' : optimizer_params, 'learning_rate_params' : learning_rate_params, 
		'what_to_save_params' : what_to_save_params, 'load_params' : load_params,
		'data_params' : data_params, 'inter_op_parallelism_threads' : inter_op_parallelism_threads,
		'allow_growth' : allow_growth, 'per_process_gpu_memory_fraction' : per_process_gpu_memory_fraction,
		'postprocessor_params' : postprocessor_params
		}


	config = tf.ConfigProto(allow_soft_placement = True,
                log_device_placement = log_device_placement, inter_op_parallelism_threads = inter_op_parallelism_threads)
	if allow_growth:
		#including this weird conditional because I'm running into a weird bug
		config.gpu_options.allow_growth = allow_growth
	if per_process_gpu_memory_fraction is not None:
		print('limiting mem fraction')
		config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
	sess = tf.Session(config = config)
	dbinterface = base.DBInterface(sess = sess, global_step = updater.global_step, params = params, save_params = save_params, load_params = load_params)
	
	dbinterface.initialize()
	data_provider.start_runner(sess)


	train(sess, updater, dbinterface)




def train(sess, updater, dbinterface):
	#big_save_keys = what_to_save_params['big_save_keys']
	#little_save_keys = what_to_save_params['little_save_keys']
	#big_save_len = what_to_save_params['big_save_len']
	#big_save_freq = what_to_save_params['big_save_freq']
	#print('big save stuff: ' + str((big_save_len, big_save_freq)))
	while True:
		dbinterface.start_time_step = time.time()
		res = updater.update(sess)
		#dividing by 2 since there are two global steps per update, as currently defined
		#global_step = res['um_global_step'] / 2
		#if (global_step - 1) % big_save_freq < big_save_len:
		#	save_keys = big_save_keys
		#else:
		#	save_keys = little_save_keys
		#print('global step, save keys ' + str((global_step, save_keys)))
		#res = dict((k, v) for k, v in res.iteritems() if k in save_keys)
		#batch_to_save = dict((k, v) for k, v in batch.iteritems() if k in save_keys)
		#res.update(batch_to_save)
		dbinterface.save(train_res = res, validation_only = False)

example_scene_local = [
        {
        'type' : 'SHAPENET',
        'scale' : 2.,
        'mass' : 1.,
        'scale_var' : .01,
        'num_items' : 1,
        }
        ]



def train_local(
		optimizer_params = None,
		learning_rate_params = None,
		model_params = None,
		visualize = False,
		data_params = None,
		exp_id = None
	):
	#set up models
	cfg = model_params['cfg']
	world_model = LatentSpaceWorldModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'])

	#saver setup
	how_often = 1500
	save_dir = os.path.join('/Users/nickhaber/Desktop/', exp_id)
	if os.path.exists(save_dir):
		if not 'test' in exp_id:
			raise Exception('Path already exists')
	else:
		os.mkdir(save_dir)

	#set up data provider
	state_memory_len = {
		STATE_DESC : 4,
	}
	rescale_dict = {
		STATE_DESC : (64, 64)
	}
	provider_params = {
		'batching_fn' : lambda hist : data.batch_FIFO(hist, batch_size = 2),
		'capacity' : 5,
		'gather_per_batch' : 2,
		'gather_at_beginning' : 4
	}

	action_to_message = lambda action, env : environment.normalized_action_to_ego_force_torque(action, env, data_params['action_limits'], wall_safety = .5)
	env = environment.Environment(1, 1, action_to_message, SCREEN_DIMS = (128, 170), USE_TDW = False, host_address = None, state_memory_len = state_memory_len, action_memory_len = 3, message_memory_len = 2, rescale_dict = rescale_dict, room_dims = (5., 5.), rng_source = environment.PeriodicRNGSource(3, seed = 1))
	scene_infos = data.SillyLittleListerator([example_scene_local])
	steps_per_scene = data.SillyLittleListerator([100])
	data_provider = data.BSInteractiveDataProvider(env, uncertainty_model, scene_infos, steps_per_scene, UniformActionSampler(cfg), ** provider_params)

	#set up updater
	postprocessor = get_default_postprocessor(what_to_save_params = LATENT_WHAT_TO_SAVE_PARAMS)
	updater = LatentUncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor)

	#do the training loop!
	sess = tf.Session()
	updater.start(sess)
	while True:
		res = updater.update(sess, visualize)
		print('updated')
		if 'batch' in res:
			depths = res['batch']['obs'][1]
			print('after: ' + str(depths.shape))
			print(np.linalg.norm(depths))
			depths = depths.astype(np.uint8)
			depths = np.concatenate([depths[:, :, 2:], depths[:, :, 1:2], depths[:, :, :1]], axis = 2)
			# orig_shape = depths.shape[:2]
			# zerosliceshape = list(orig_shape) + [2]
			# chan1 = np.copy(depths)
			# chan1[:, :, 1:] = np.zeros(zerosliceshape)
			# chan2 = np.copy(depths)
			# chan2[:, :, 0] = np.zeros(orig_shape)
			# chan2[:, :, 2] = np.zeros(orig_shape)
			# chan3 = np.copy(depths)
			# chan3[:, :, :2] = np.zeros(zerosliceshape)
			cv2.imshow('view', depths.astype(np.uint8))
			# cv2.imshow('1', chan1)
			# cv2.imshow('2', chan2)
			# cv2.imshow('3', chan3)
			# print('diffs')
			# print(np.linalg.norm(depths[:, :, 0] - depths[:, :, 1]))
			# print(np.linalg.norm(depths[:, :, 1] - depths[:, :, 2]))
			cv2.waitKey(1)
			print(res['msg'])
			print(res['batch']['act_post'])
		# saver.update(res)









