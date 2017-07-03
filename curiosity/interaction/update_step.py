'''
Defines the training step.
'''


import sys
sys.path.append('tfutils')
import tensorflow as tf
from tfutils.base import get_optimizer, get_learning_rate
import numpy as np
import cv2
from curiosity.interaction import models



class RawDepthDiscreteActionUpdater:
	'''
	Provides the training step.
	This is probably where we can put parallelization.
	Not finished!
	'''
	def __init__(world_model, rl_model, data_provider, eta):
		self.data_provider = data_provider
		self.world_model = world_model
		self.rl_model = rl_model
		self.eta = eta
		self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))

		self.action = tf.placeholder = tf.placeholder(tf.float32, [None] + world_model.action_one_hot.get_shape().as_list()[1:])
		self.adv = tf.placeholder(tf.float32, [None])
		self.r = tf.placeholder(tf.float32, [None])

		log_prob_tf = tf.nn.log_softmax(rl_model.logits)
		prob_tf = tf.nn.softmax(rl_model.logits)
		pi_loss = -tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)
		vf_loss = .5 * tf.reduce_sum(tf.square(rl_model.vf - self.r))
		entropy = -tf.reduce_sum(prob_tf * log_prob_tf)

		self.rl_loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

		rl_opt_params, rl_opt = get_optimizer(learning_rate, self.rl_loss, )


def replace_the_nones(my_list):
	'''
		Assumes my_list[-1] is np array
	'''
	return [np.zeros(my_list[-1].shape, dtype = my_list[-1].dtype) if elt is None else elt for elt in my_list]


def postprocess_batch_depth(batch):
	obs, msg, act = batch
	depths = replace_the_nones(obs['depths1'])
	obs_past = np.array([depths[:-1]])
	obs_fut = np.array([depths[1:]])
	actions = np.array([replace_the_nones(act)])
	return obs_past, actions, obs_fut




# def postprocess_batch_depth(batch):
# 	depths = np.array([[timepoint if timepoint is not None else np.zeros(obs['depths1'][-1].shape, dtype = obs['depths1'][-1].dtype) for timepoint in obs['depths1']] for obs in batch.states])
# 	actions = np.array(batch.actions)
# 	next_depth =  np.array([batch.next_state['depths1']])
# 	return depths, actions, next_depth


def postprocess_batch_for_actionmap(batch):
	obs, msg, act = batch
	prepped = {}
	for desc in ['depths1', 'objects1']:
		dat = obs[desc]


def postprocess_batch_for_actionmap(batch):
	prepped = {}
	for desc in ['depths1', 'objects1']:
		prepped[desc] = np.array([[timepoint if timepoint is not None else np.zeros(obs[desc][-1].shape, dtype = obs[desc][-1].dtype) for timepoint in obs[desc]] for obs in batch.states])
	actions = np.array([ for ])
	actions = np.array([[np.zeros(batch.next_state['action'][-1].shape, batch.next_state['action'][-1].dtype) if timepoint is None else timepoint for timepoint in batch.next_state['action']]])
	print('actions shape')
	print(actions.shape)
	print(len(batch.next_state['action']))
	action_ids_list = []
	for i in range(2):
		action_msg = batch.next_state['msg'][i]['msg']['actions'] if batch.next_state['msg'][i] is not None else []
		if len(action_msg):
			idx = int(action_msg[0]['id'])
		else:
			idx = -10000
		action_ids_list.append(idx)
	action_ids = np.array([action_ids_list])
	next_depths =  np.array([batch.next_state['depths1']])
	return prepped['depths1'], prepped['objects1'], actions, action_ids, next_depths

class UncertaintyPostprocessor:
	def __init__(self, big_save_keys, little_save_keys, big_save_len, big_save_freq):
		self.big_save_keys = big_save_keys
		self.little_save_keys = little_save_keys
		self.big_save_len = big_save_len
		self.big_save_freq = big_save_freq

	def postprocess(self, training_results, batch):
		global_step = training_results['um_global_step'] / 2
		if (global_step - 1) % self.big_save_freq < self.big_save_len:
			save_keys = self.big_save_keys
		else:
			save_keys = self.little_save_keys
		res = dict((k, v) for (k, v) in training_results.iteritems() if k in save_keys)
		res['msg'] = batch.next_state['msg']
		return res

class UncertaintyUpdater:
	def __init__(self, world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor):
		self.data_provider = data_provider
		self.world_model = world_model
		self.um = uncertainty_model
		self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
		self.wm_lr_params, wm_learning_rate = get_learning_rate(self.global_step, ** learning_rate_params['world_model'])
		self.wm_opt_params, wm_opt = get_optimizer(wm_learning_rate, self.world_model.loss, self.global_step, optimizer_params['world_model'])
		self.world_model_targets = {'given' : self.world_model.processed_input,  'loss' : self.world_model.loss, 'learning_rate' : wm_learning_rate, 'optimizer' : wm_opt, 'prediction' : self.world_model.pred, 'tv' : self.world_model.tv}
		self.inc_step = self.global_step.assign_add(1)
		self.wm_lr_params, um_learning_rate = get_learning_rate(self.global_step, **learning_rate_params['uncertainty_model'])
		self.wm_lr_params, um_opt = get_optimizer(um_learning_rate, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'])
		self.um_targets = {'loss' : self.um.uncertainty_loss, 'learning_rate' : um_learning_rate, 'optimizer' : um_opt, 'global_step' : self.global_step}
		self.postprocessor = postprocessor

	def start(self, sess):
		self.data_provider.start_runner(sess)
		sess.run(tf.global_variables_initializer())

	def update(self, sess, visualize = False):
		batch = self.data_provider.dequeue_batch()
		depths, actions, next_depth = postprocess_batch_depth(batch)
		wm_feed_dict = {
			self.world_model.s_i : depths,
			self.world_model.s_f : next_depth,
			self.world_model.action : actions
		}
		world_model_res = sess.run(self.world_model_targets, feed_dict = wm_feed_dict)
		if visualize:
			cv2.imshow('pred', world_model_res['prediction'][0] / 4.)#TODO clean up w colors
			cv2.imshow('tv', world_model_res['tv'][0] / 4.)
			cv2.imshow('processed0', world_model_res['given'][0, 0] / 4.)
			cv2.imshow('processed1', world_model_res['given'][0, 1] / 4.)
			cv2.waitKey(1)
			print('wm loss: ' + str(world_model_res['loss']))
		um_feed_dict = {
			self.um.s_i : depths,
			self.um.action_sample : actions,
			self.um.true_loss : np.array([world_model_res['loss']])
		}
		um_res = sess.run(self.um_targets, feed_dict = um_feed_dict)
		wm_res_new = dict(('wm_' + k, v) for k, v in world_model_res.iteritems())
		um_res_new = dict(('um_' + k, v) for k, v in um_res.iteritems())
		wm_res_new.update(um_res_new)
		res = self.postprocessor.postprocess(wm_res_new, batch)
		return res



class DamianWMUncertaintyUpdater:
	def __init__(self, world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor):
		self.data_provider = data_provider
		self.world_model = world_model
		self.um = uncertainty_model
		self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
		self.wm_lr_params, wm_learning_rate = get_learning_rate(self.global_step, ** learning_rate_params['world_model'])
		self.wm_opt_params, wm_opt = get_optimizer(wm_learning_rate, self.world_model.loss, self.global_step, optimizer_params['world_model'])
		self.world_model_targets = {'given' : self.world_model.processed_input,  'loss' : self.world_model.loss, 'learning_rate' : wm_learning_rate, 'optimizer' : wm_opt, 'prediction' : self.world_model.pred, 'tv' : self.world_model.tv}
		self.inc_step = self.global_step.assign_add(1)
		self.wm_lr_params, um_learning_rate = get_learning_rate(self.global_step, **learning_rate_params['uncertainty_model'])
		self.wm_lr_params, um_opt = get_optimizer(um_learning_rate, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'])
		self.um_targets = {'loss' : self.um.uncertainty_loss, 'learning_rate' : um_learning_rate, 'optimizer' : um_opt, 'global_step' : self.global_step}
		self.postprocessor = postprocessor

	def start(self, sess):
		self.data_provider.start_runner(sess)
		sess.run(tf.global_variables_initializer())

	def update(self, sess, visualize = False):
		batch = self.data_provider.dequeue_batch()
		depths, objects, actions, action_ids, next_depth = postprocess_batch_for_actionmap(batch)
		wm_feed_dict = {
			self.world_model.s_i : depths,
			self.world_model.s_f : next_depth,
			self.world_model.action : actions,
			self.world_model.action_id : action_ids, 
			self.world_model.objects : objects
		}
		world_model_res = sess.run(self.world_model_targets, feed_dict = wm_feed_dict)
		if visualize:
			cv2.imshow('pred', world_model_res['prediction'][0] / 4.)#TODO clean up w colors
			cv2.imshow('tv', world_model_res['tv'][0] / 4.)
			cv2.imshow('processed0', world_model_res['given'][0, 0] / 4.)
			cv2.imshow('processed1', world_model_res['given'][0, 1] / 4.)
			cv2.waitKey(1)
			print('wm loss: ' + str(world_model_res['loss']))
		um_feed_dict = {
			self.um.s_i : depths,
			self.um.action_sample : actions[:, -1],
			self.um.true_loss : np.array([world_model_res['loss']])
		}
		um_res = sess.run(self.um_targets, feed_dict = um_feed_dict)
		wm_res_new = dict(('wm_' + k, v) for k, v in world_model_res.iteritems())
		um_res_new = dict(('um_' + k, v) for k, v in um_res.iteritems())
		wm_res_new.update(um_res_new)
		res = self.postprocessor.postprocess(wm_res_new, batch)
		return res











