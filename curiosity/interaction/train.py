'''
train.py
has the actual loop for training, given all the parameters and things.
This should eventually be like tfutils.base, though for now we specialize a lot.
'''

import curiosity.interaction.environment as environment
import curiosity.interaction.data as data
from curiosity.interaction.data import SimpleSamplingInteractiveDataProvider, SillyLittleListerator
from curiosity.interaction.models import UncertaintyModel, DepthFuturePredictionWorldModel, UniformActionSampler, DamianModel
from curiosity.interaction.update_step import UncertaintyUpdater, UncertaintyPostprocessor, DamianWMUncertaintyUpdater
import tensorflow as tf
import os
import cPickle
import tfutils.base as base
import time

RENDER_2_ADDY = '10.102.2.162'

class LocalSaver:
	def __init__(self, how_much, how_often, save_dir):
		self.how_much = how_much
		self.how_often = how_often
		self.save_dir = save_dir
		self.storage = dict((k, []) for k in how_much)
		self.ctr = 0
		with open(os.path.join(self.save_dir, 'test.pkl'), 'w') as stream:
			cPickle.dump([], stream)

	def update(self, results):
		for k in self.how_much:
			if len(self.storage[k]) < self.how_much[k]:
				self.storage[k].append(results[k])
		self.ctr += 1
		if self.ctr % self.how_often == 0:
			print('saving...')
			fn = os.path.join(self.save_dir, 'sv_' + str(self.ctr // self.how_often) + '.pkl')
			with open(fn, 'w') as stream:
				cPickle.dump(self.storage, stream)
				self.storage = dict((k, []) for k in self.how_much)


def get_default_postprocessor(what_to_save_params):
	return UncertaintyPostprocessor(** what_to_save_params)


def get_models_damianworld(cfg):
	world_model = DamianModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'])
	return {'world_model' : world_model, 'uncertainty_model' : uncertainty_model}


def get_default_models(cfg):
	world_model = DepthFuturePredictionWorldModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'])
	return {'world_model' : world_model, 'uncertainty_model' : uncertainty_model}

def get_default_data_provider(data_params, model_params, action_model):
	action_to_message = lambda action, env : environment.normalized_action_to_ego_force_torque(action, env, data_params['action_limits'], wall_safety = .5)
	env = environment.Environment(action_to_message_fn = action_to_message, ** data_params['environment_params'])
	scene_infos = data.SillyLittleListerator(data_params['scene_list'])
	steps_per_scene = data.SillyLittleListerator(data_params['scene_lengths'])
	data_provider = SimpleSamplingInteractiveDataProvider(env, action_model, 1, scene_infos, steps_per_scene, UniformActionSampler(model_params['cfg']), data_params['capacity'])
	return data_provider

def get_default_updater(models, data_provider, optimizer_params, learning_rate_params, postprocessor):
	world_model = models['world_model']
	uncertainty_model = models['uncertainty_model']
	return UncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor)

def get_damian_updater(models, data_provider, optimizer_params, learning_rate_params, postprocessor):
	world_model = models['world_model']
        uncertainty_model = models['uncertainty_model']
        return DamianWMUncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor)


DEFAULT_WHAT_TO_SAVE_PARAMS = {
		'big_save_keys' : ['um_loss', 'wm_loss', 'wm_prediction', 'wm_tv', 'wm_given'],
		'little_save_keys' : ['um_loss', 'wm_loss'],
		'big_save_len' : 100,
		'big_save_freq' : 10000
	}




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
		inter_op_parallelism_threads = 40
	):
	model_cfg = model_params['cfg']
	models_constructor = model_params['func']
	models = models_constructor(model_cfg)

	action_model = models[model_params['action_model_desc']]

	data_provider = data_params['func'](data_params, model_params, action_model)

	postprocessor = get_default_postprocessor(what_to_save_params)

	updater = train_params['updater_func'](models, data_provider, optimizer_params, learning_rate_params, postprocessor)

	params = {'save_params' : save_params, 'model_params' : model_params, 'train_params' : train_params, 'optimizer_params' : optimizer_params, 'learning_rate_params' : learning_rate_params, 'what_to_save_params' : what_to_save_params, 'load_params' : load_params}

	sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, 
		log_device_placement = log_device_placement, inter_op_parallelism_threads = inter_op_parallelism_threads))
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
	world_model = DepthFuturePredictionWorldModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'])

	#saver setup
	how_often = 1500
	save_dir = os.path.join('/Users/nickhaber/Desktop/', exp_id)
	if os.path.exists(save_dir):
		raise Exception('Path already exists')
	else:
		os.mkdir(save_dir)

	how_much = {'um_loss' : 1500, 'wm_loss' : 1500, 'wm_prediction' : 50, 'wm_tv' : 50, 'wm_given' : 50}
	saver = LocalSaver(how_much, how_often, save_dir)

	#set up data provider
	state_memory_len = {
		'depths1' : 3
	}
	rescale_dict = {
		'depths1' : (64, 64)
	}
	action_to_message = lambda action, env : environment.normalized_action_to_ego_force_torque(action, env, data_params['action_limits'], wall_safety = .5)
	env = environment.Environment(1, 1, action_to_message, USE_TDW = True, host_address = RENDER_2_ADDY, state_memory_len = state_memory_len, rescale_dict = rescale_dict, room_dims = (5., 5.))
	scene_infos = data.SillyLittleListerator([environment.example_scene_info])
	steps_per_scene = data.SillyLittleListerator([1024 * 32])
	data_provider = SimpleSamplingInteractiveDataProvider(env, uncertainty_model, 1, scene_infos, steps_per_scene, UniformActionSampler(cfg), capacity = 5)

	#set up updater
	postprocessor = get_default_postprocessor(what_to_save_params = DEFAULT_WHAT_TO_SAVE_PARAMS)
	updater = UncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor)

	#do the training loop!
	sess = tf.Session()
	updater.start(sess)
	while True:
		res = updater.update(sess, visualize)
		saver.update(res)









