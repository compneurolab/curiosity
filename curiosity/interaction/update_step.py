'''
Defines the training step.
'''


import sys
sys.path.append('tfutils')
import tensorflow as tf
from tfutils.base import get_optimizer, get_learning_rate
import numpy as np
import cv2



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



def postprocess_batch_depth(batch):
	depths = np.array([obs['depth'] for obs in batch.states])
	actions = np.array(batch.actions)
	next_depth =  np.array([batch.next_state['depth']])
	return depths, actions, next_depth

class UncertaintyUpdater:
	def __init__(self, world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params):
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
		self.um_targets = {'loss' : self.um.uncertainty_loss, 'learning_rate' : um_learning_rate, 'optimizer' : um_opt}

	def start(self, sess):
		self.data_provider.start_runner(sess)
		sess.run(tf.global_variables_initializer())

	def update(self, sess, visualize):
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
		return world_model_res, um_res













