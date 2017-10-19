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
import h5py
import json


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


def postprocess_batch_depth(batch, state_desc):
	obs, msg, act, act_post = batch
	depths = replace_the_nones(obs[state_desc])
	obs_past = np.array([depths[:-1]])
	obs_fut = np.array([depths[1:]])
	actions = np.array([replace_the_nones(act)])
	actions_post = np.array([replace_the_nones(act_post)])
	return obs_past, actions, actions_post, obs_fut




# def postprocess_batch_depth(batch):
# 	depths = np.array([[timepoint if timepoint is not None else np.zeros(obs['depths1'][-1].shape, dtype = obs['depths1'][-1].dtype) for timepoint in obs['depths1']] for obs in batch.states])
# 	actions = np.array(batch.actions)
# 	next_depth =  np.array([batch.next_state['depths1']])
# 	return depths, actions, next_depth


def postprocess_batch_for_actionmap(batch, state_desc):
	obs, msg, act = batch
	prepped = {}
	depths = replace_the_nones(obs[state_desc])
	depths_past = np.array([depths[:-1]])
	depths_fut = np.array([depths[:1]])
	objects = np.array([replace_the_nones(obs[state_desc])[:-1]])
	actions = np.array([replace_the_nones(act)])
	action_ids_list = []
	for i in range(2):
		action_msg = msg[i]['msg']['actions'] if msg[i] is not None else []
		if len(action_msg):
			idx = int(action_msg[0]['id'])
		else:
			idx = -10000#just something that's not an id seen
		action_ids_list.append(idx)
	action_ids = np.array([action_ids_list])
	return  depths_past, objects, actions, action_ids, depths_fut


# def postprocess_batch_for_actionmap(batch):
# 	prepped = {}
# 	for desc in ['depths1', 'objects1']:
# 		prepped[desc] = np.array([[timepoint if timepoint is not None else np.zeros(obs[desc][-1].shape, dtype = obs[desc][-1].dtype) for timepoint in obs[desc]] for obs in batch.states])
# 	actions = np.array([[np.zeros(batch.next_state['action'][-1].shape, batch.next_state['action'][-1].dtype) if timepoint is None else timepoint for timepoint in batch.next_state['action']]])
# 	print('actions shape')
# 	print(actions.shape)
# 	print(len(batch.next_state['action']))
# 	action_ids_list = []
# 	for i in range(2):
# 		action_msg = batch.next_state['msg'][i]['msg']['actions'] if batch.next_state['msg'][i] is not None else []
# 		if len(action_msg):
# 			idx = int(action_msg[0]['id'])

# 		action_ids_list.append(idx)
# 	action_ids = np.array([action_ids_list])
# 	next_depths =  np.array([batch.next_state['depths1']])
# 	return prepped['depths1'], prepped['objects1'], actions, action_ids, next_depths

class ExperienceReplayPostprocessor:
	def __init__(self, big_save_keys = None, little_save_keys = None, big_save_len = None, big_save_freq = None, state_descriptor = None):
		self.big_save_keys = big_save_keys
		self.little_save_keys = little_save_keys
		self.big_save_len = big_save_len
		self.big_save_freq = big_save_freq
		self.state_descriptor = state_descriptor
                self.big_save_keys.append('map_draw')
                self.little_save_keys.append('map_draw')
                self.big_save_keys.extend(['act_lr', 'um_lr'])
                self.little_save_keys.extend(['act_lr', 'um_lr'])


	def postprocess(self, training_results, batch):
		global_step = training_results['global_step']
		res = {}
		if (global_step) % self.big_save_freq < self.big_save_len:
			save_keys = self.big_save_keys
			#est_losses = [other[1] for other in batch['other']]
			#action_sample = [other[2] for other in batch['other']]
			res['batch'] = {}
			for desc, val in batch.iteritems():
			    if desc not in  ['recent', 'depths1']:
			        res['batch'][desc] = val
			res['recent'] = batch['recent']
		else:
			save_keys = self.little_save_keys
                res.update(dict(pair for pair in training_results.iteritems() if pair[0] in save_keys))
		#if 'other' in batch['recent']:
		#	entropies = [other[0] for other in batch['recent']['other']]
               # 	entropies = np.mean(entropies)
               # 	res['entropy'] = entropies
		if 'msg' in batch['recent']:
			looking_at_obj = [1 if msg is not None and msg['msg']['action_type'] == 'OBJ_ACT' else 0 for msg in batch['recent']['msg']]
                	res['obj_freq'] = np.mean(looking_at_obj)
                elif type(batch['recent']) == list:
                    mean_per_provider = []
                    for provider_recent in batch['recent']:
                        looking_at_obj = [1 if msg is not None and msg['msg']['action_type'] == 'OBJ_ACT' else 0 for msg in provider_recent['msg']]
                        mean_per_provider.append(np.mean(looking_at_obj))
                    res['obj_freq'] = np.mean(looking_at_obj)
                return res

class UncertaintyPostprocessor:
	def __init__(self, big_save_keys = None, little_save_keys = None, big_save_len = None, big_save_freq = None, state_descriptor = None):
		self.big_save_keys = big_save_keys
		self.little_save_keys = little_save_keys
		self.big_save_len = big_save_len
		self.big_save_freq = big_save_freq
		self.state_descriptor = state_descriptor

	def postprocess(self, training_results, batch):
		global_step = training_results['global_step']
		res = {}
		print('postprocessor deets')
		print(global_step)
		print(self.big_save_freq)
		print(self.big_save_len)
		if (global_step) % self.big_save_freq < self.big_save_len:
			print('big time')
			save_keys = self.big_save_keys
			est_losses = [other[1] for other in batch['recent']['other']]
			action_sample = [other[2] for other in batch['recent']['other']]
			res['batch'] = {'obs' : batch['depths1'], 'act' : batch['action'], 'act_post' : batch['action_post'],  'est_loss' : est_losses, 'action_sample' : action_sample}
			res['msg'] = batch['recent']['msg']
		else:
			print('little time')
			save_keys = self.little_save_keys
		res.update(dict((k, v) for (k, v) in training_results.iteritems() if k in save_keys))
		#res['msg'] = batch['msg'][-1]
		entropies = [other[0] for other in batch['recent']['other']]
		entropies = np.mean(entropies)
		res['entropy'] = entropies
		looking_at_obj = [1 if msg is not None and msg['msg']['action_type']['OBJ_ACT'] else 0 for msg in batch['recent']['msg']]
		res['obj_freq'] = np.mean(looking_at_obj)
		return res


class DataWriteUpdater:
	def __init__(self, data_provider, updater_params):
		self.data_provider = data_provider
		fn = updater_params['hdf5_filename']
		N = updater_params['N_save']
		height, width = updater_params['image_shape']
		act_dim = updater_params['act_dim']
		print('setting up save loc')
		self.hdf5 = hdf5 = h5py.File(fn, mode = 'a')
		dt = h5py.special_dtype(vlen = str)
		self.handles = {'msg' : hdf5.require_dataset('msg', shape = (N,), dtype = dt),
				'depths1' : hdf5.require_dataset('depths1', shape = (N, height, width, 3), dtype = np.uint8),
				'action' : hdf5.require_dataset('action', shape = (N, act_dim), dtype = np.float32),
				'action_post' : hdf5.require_dataset('action_post', shape = (N, act_dim), dtype = np.float32)}
		print('save loc set up')
		self.start = 0	

	def update(self):
		batch = self.data_provider.dequeue_batch()
		bs = len(batch['recent']['msg'])
		end = self.start + bs
		for k in ['depths1', 'action', 'action_post']:
			tosave = batch['recent'][k]
			if k in ['action', 'action_post']:
				tosave = tosave.astype(np.float32)
			self.handles[k][self.start : end] = batch['recent'][k]
		self.handles['msg'][self.start : end] = [json.dumps(msg) for msg in batch['recent']['msg']]
		self.start = end

	def close(self):
		self.hdf5.close()			


class LatentUncertaintyValidator:
	def __init__(self, models, data_provider):
		self.um = models['uncertainty_model']
		self.wm = models['world_model']
		self.targets = {
						'act_pred' : self.wm.act_pred, 
						'fut_loss' : self.wm.fut_loss, 'act_loss' : self.wm.act_loss, 'um_loss' : self.um.uncertainty_loss,
						'estimated_world_loss' : self.um.estimated_world_loss, 'loss_per_example' : self.um.true_loss,
						'act_loss_per_example' : self.wm.act_loss_per_example
						}
		self.dp = data_provider

	def run(self, sess):
		batch = self.dp.dequeue_batch()
		feed_dict = {
						self.wm.states : batch['depths1'],
						self.wm.action : batch['action'],
						self.wm.action_post : batch['action_post'],
						self.wm.obj_there : batch['obj_there']
				}
		res = sess.run(self.targets, feed_dict = feed_dict)
		res['batch'] = {}
		for desc, val in batch.iteritems():
			print(desc)
			if desc == 'obj_there':
				res['batch'][desc] = val
			elif desc != 'recent':
				res['batch'][desc] = val[:, -1]
		res['recent'] = batch['recent']






class ObjectThereValidater:
	def __init__(self, models, data_provider):
		self.um = models['uncertainty_model']
		self.wm = models['world_model']
		self.targets = {'um_loss' : self.um.uncertainty_loss, 'loss_per_example' : self.um.true_loss,
				'estimated_world_loss' : self.um.estimated_world_loss}
		self.dp = data_provider

	def run(self, sess):
		batch = self.dp.dequeue_batch()
                feed_dict = {
                        self.wm.states : batch['depths1'],
                        self.wm.action : batch['action'],
                        self.wm.obj_there : batch['obj_there']
                }
		return sess.run(self.targets, feed_dict = feed_dict)


class ActionUncertaintyValidator:
	def __init__(self, models, data_provider):
		self.um = um = models['uncertainty_model']
		self.wm = wm = models['world_model']
		self.targets = {'act_pred' : self.wm.act_pred, 'act_loss' : self.wm.act_loss,
                                'estimated_world_loss' : self.um.estimated_world_loss,
                                'um_loss' : self.um.uncertainty_loss, 'loss_per_example' : self.um.true_loss}
		self.dp = data_provider
	
	def run(self, sess):
		batch = self.dp.dequeue_batch()
		feed_dict = {
			self.wm.states : batch['depths1'],
			self.wm.action : batch['action'],
			self.wm.action_post : batch['action_post']
		}
		res = sess.run(self.targets, feed_dict = feed_dict)
		res['batch'] = batch
		return res


class ActionUncertaintyValidatorWithReadouts:
    def __init__(self, model, data_provider):
        self.dp = data_provider
        self.wm = model['world_model']
        self.um = model['uncertainty_model']
        self.targets = {}
        self.targets.update(self.wm.readouts)
        self.targets.update(self.um.readouts)
        #this should be changed for an online data provider, set to do nothing
        self.map_draw_mode = 'specified_indices'
        #TODO read this off other stuff
        hardcoded_cfg = {'world_model' : {'act_dim' : 5}, 'uncertainty_model' : {'n_action_samples' : 1000}, 'seed' : 0}
        self.action_sampler = models.UniformActionSampler(hardcoded_cfg)
        self.map_draw_example_indices = [0, 31]
        self.map_draw_timestep_indices = [1, 2]
        self.state_desc = 'depths1'

    def run(self, sess):
        batch = self.dp.dequeue_batch()
        feed_dict = {
                self.wm.states : batch['depths1'],
                self.wm.action : batch['action'],
                self.wm.action_post : batch ['action_post']
                }
        res = sess.run(self.targets, feed_dict = feed_dict)
        #TODO case it for online
        res['recent'] = {}
        if self.map_draw_mode == 'specified_indices':
            map_draw_res = []
            for idx in self.map_draw_example_indices:
                obs_for_actor = [batch[self.state_desc][idx][t] for t in self.map_draw_timestep_indices]
                action_samples = self.action_sampler.sample_actions()
                action, entropy, estimated_world_loss = self.um.act(sess, action_samples, obs_for_actor)
                to_add = {'example_id' : idx, 'action_sample' : action, 'estimated_world_loss' : estimated_world_loss,
                        'action_samples' : action_samples, 'depths1' : batch[self.state_desc][idx],
                        'action' : batch['action'][idx], 'action_post' : batch['action_post'][idx]}
                map_draw_res.append(to_add)
        res['map_draw'] = map_draw_res
        return res





class ObjectThereUpdater:
	def __init__(self, world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params):
		self.data_provider = data_provider
		self.wm = world_model
		self.um = uncertainty_model
		self.postprocessor = postprocessor
		self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
		self.um_lr_params, um_lr = get_learning_rate(self.global_step, ** learning_rate_params['uncertainty_model'])
		um_opt_params, um_opt = get_optimizer(um_lr, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'], var_list = self.um.var_list)
		self.targets = {'um_loss' : self.um.uncertainty_loss, 'um_lr' : um_lr, 'um_optimizer' : um_opt, 
						'global_step' : self.global_step, 'loss_per_example' : self.um.true_loss,
						'estimated_world_loss' : self.um.estimated_world_loss
								}
		self.state_desc = updater_params['state_desc']


	def update(self, sess, visualize = False):
		batch = self.data_provider.dequeue_batch()
		state_desc = self.state_desc
		feed_dict = {
			self.wm.states : batch[state_desc],
			self.wm.action : batch['action'],
			self.wm.obj_there : batch['obj_there']
		}
		res = sess.run(self.targets, feed_dict = feed_dict)
		res = self.postprocessor.postprocess(res, batch)
		return res


class SquareForceMagUpdater:
	def __init__(self, models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params):
		self.dp = data_provider
		self.wm = models['world_model']
		self.um = models['uncertainty_model']
		self.postprocessor = postprocessor
                self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
                self.um_lr_params, um_lr = get_learning_rate(self.global_step, ** learning_rate_params['uncertainty_model'])
                um_opt_params, um_opt = get_optimizer(um_lr, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'], var_list = self.um.var_list)
		self.targets = {'um_loss' : self.um.uncertainty_loss, 'um_lr' : um_lr, 'um_optimizer' : um_opt,
                                                'global_step' : self.global_step, 'loss_per_example' : self.um.true_loss,
                                                'estimated_world_loss' : self.um.estimated_world_loss
                                                                }
		if self.um.exactly_whats_needed:
			self.targets['oh_my_god'] = self.um.oh_my_god
		self.state_desc = updater_params['state_desc']
		
	
	def update(self, sess, visualize = False):
		batch = self.dp.dequeue_batch()
		state_desc = self.state_desc
		feed_dict = {
			self.wm.states : batch[state_desc],
			self.wm.action : batch['action'],
			self.wm.action_post : batch['action_post']
		}
		if self.um.insert_obj_there:
			print('adding obj_there to feed dict')
			feed_dict[self.um.obj_there] = batch['obj_there']
		res = sess.run(self.targets, feed_dict = feed_dict)
		res = self.postprocessor.postprocess(res, batch)
		return res


class DebuggingForceMagUpdater:
	def __init__(self, models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params):
		self.dp = data_provider
		self.wm = models['world_model']
		self.um = models['uncertainty_model']
		self.postprocessor = postprocessor
		self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0, dtype = tf.int32))
		print(learning_rate_params.keys())
		um_lr_params, um_lr = get_learning_rate(self.global_step, **learning_rate_params['uncertainty_model'])
		um_opt_params, um_opt = get_optimizer(um_lr, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'])
		self.targets = {'um_loss' : self.um.uncertainty_loss, 'um_optimizer' : um_opt, 'global_step' : self.global_step,
				'loss_per_example' : self.um.true_loss, 'estimated_world_loss' : self.um.estimated_world_loss, 'ans' : self.um.ans,
				'oh_my_god' : self.um.oh_my_god, 'model_parameters' : self.um.var_list}
	
	def update(self, sess):
		batch = self.dp.dequeue_batch()
		feed_dict = {
			self.wm.action : batch['action'],
			self.wm.action_post : batch['action_post'],
			self.um.obj_there : batch['obj_there']
		}
		res = sess.run(self.targets, feed_dict = feed_dict)
		res = self.postprocessor.postprocess(res, batch)
		return res





class FreezeUpdater:
    def __init__(self, models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params):
        self.data_provider = data_provider \
                if isinstance(data_provider, list) else [data_provider]
        self.wm = models['world_model']
        self.um = models['uncertainty_model']
        freeze_wm = updater_params['freeze_wm']
        freeze_um = updater_params['freeze_um']
        self.postprocessor = postprocessor
        self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
        self.act_lr_params, act_lr = get_learning_rate(self.global_step, ** learning_rate_params['world_model']['act_model'])
        self.um_lr_params, um_lr = get_learning_rate(self.global_step, ** learning_rate_params['uncertainty_model'])
        num_not_frozen = 0
        self.targets = {}
        self.state_desc = 'depths1'
        if not freeze_wm:
            num_not_frozen += 1
            act_opt_params, act_opt = get_optimizer(act_lr, self.wm.act_loss, self.global_step, optimizer_params['world_model']['act_model'], var_list = self.wm.act_var_list + self.wm.encode_var_list)
            self.targets['act_opt'] = act_opt
        if not freeze_um:
            num_not_frozen += 1
            um_opt_params, um_opt = get_optimizer(um_lr, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'], var_list = self.um.var_list)
            self.targets['um_opt'] = um_opt
        if num_not_frozen == 0:
            self.targets['global_step'] = self.global_step
            self.targets['increment'] = tf.assign_add(self.global_step, 1)
        else:
            self.global_step = self.global_step / num_not_frozen
            self.targets['global_step'] = self.global_step
            self.targets.update({'act_lr' : act_lr, 'um_lr' : um_lr})


        assert set(self.wm.readouts.keys()) != set(self.um.readouts.keys())
        self.targets.update(self.wm.readouts)
        self.targets.update(self.um.readouts)
        um_increment = tf.assign_add(um.step, 1)
        assert 'um_increment' not in self.targets
        self.targets['um_increment'] = um_incrememt

        #self.map_draw_mode = None
        #Map drawing. Meant to have options, but for now just assuming one sort of specification
        #self.state_desc = updater_params.get('state_desc', 'depths1')
        #self.map_draw_mode = updater_params['map_draw_mode']
        #this specification specifices batch example indices for which we do a forward pass.
        #need to do one forward pass each index because action sampling is the 'batch.'
        #self.action_sampler = action_sampler
        #assert self.map_draw_mode == 'specified_indices' and self.action_sampler is not None, (self.map_draw_mode, action_sampler)
        #self.map_draw_example_indices = updater_params['map_draw_example_indices']
        #self.map_draw_timestep_indices = updater_params['map_draw_timestep_indices']
        #self.map_draw_freq = updater_params['map_draw_freq']

        

    def update(self, sess, visualize = False):
        batch = {}
        for i, dp in enumerate(self.data_provider):
            provider_batch = dp.dequeue_batch()
            for k in provider_batch:
                if k in batch:
                    batch[k].append(provider_batch[k])
                else:
                    batch[k] = [provider_batch[k]]
        for k in ['action', 'action_post', 'depths1']:
            batch[k] = np.concatenate(batch[k], axis=0)
        feed_dict = {
                self.wm.states : batch[self.state_desc],
                self.wm.action : batch['action'],
                self.wm.action_post : batch['action_post']
            }
        res = sess.run(self.targets, feed_dict = feed_dict)
        res.pop('um_increment')
        global_step = res['global_step']
        #if self.map_draw_mode is not None and global_step % self.map_draw_freq == 0:
        #    if self.map_draw_mode == 'specified_indices':
        #        map_draw_res = []
        #        for idx in self.map_draw_example_indices:
        #            obs_for_actor = [batch[self.state_desc][idx][t] for t in self.map_draw_timestep_indices]
        #            action_samples = self.action_sampler.sample_actions()
        #            action, entropy, estimated_world_loss = self.um.act(sess, action_samples, obs_for_actor)
        #            to_add = {'example_id' : idx, 'action_sample' : action, 'estimated_world_loss' : estimated_world_loss, 
        #                    'action_samples' : action_samples, 'depths1' : batch[self.state_desc][idx], 
        #                    'action' : batch['action'][idx], 'action_post' : batch['action_post'][idx]}
        #            map_draw_res.append(to_add)
        #    res['map_draw'] = map_draw_res
        res = self.postprocessor.postprocess(res, batch)
        return res, global_step








class JustUncertaintyUpdater:
    def __init__(self, models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params, action_sampler = None):
        self.data_provider = data_provider \
                if isinstance(data_provider, list) else [data_provider]
        self.wm = models['world_model']
        self.um = models['uncertainty_model']
        self.postprocessor = postprocessor
        self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
        self.um_lr_params, um_lr = get_learning_rate(self.global_step, ** learning_rate_params['uncertainty_model'])
        um_opt_params, um_opt = get_optimizer(um_lr, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'], var_list = self.um.var_list)
        self.targets = {'global_step' : self.global_step, 'um_optimizer' : um_opt}
        assert set(self.wm.readouts.keys()) != set(self.um.readouts.keys())
        self.targets.update(self.wm.readouts)
        self.targets.update(self.um.readouts)
        #self.targets = {
        #				'fut_pred' : self.wm.fut_pred, 'act_pred' : self.wm.act_pred, 
        #				'fut_loss' : self.wm.fut_loss, 'act_loss' : self.wm.act_loss,
        #				'estimated_world_loss' : self.um.estimated_world_loss,
        #                                ''
        #				}
        #self.targets.update({'um_loss' : self.um.uncertainty_loss, 'um_lr' : um_lr, 'um_optimizer' : um_opt, 
        #				'global_step' : self.global_step, 'loss_per_example' : self.um.true_loss})
        self.map_draw_mode = None
        #Map drawing. Meant to have options, but for now just assuming one sort of specification
        self.state_desc = updater_params.get('state_desc', 'depths1')
        self.map_draw_mode = updater_params['map_draw_mode']
        #this specification specifices batch example indices for which we do a forward pass.
        #need to do one forward pass each index because action sampling is the 'batch.'
        self.action_sampler = action_sampler
        assert self.map_draw_mode == 'specified_indices' and self.action_sampler is not None, (self.map_draw_mode, action_sampler)
        self.map_draw_example_indices = updater_params['map_draw_example_indices']
        self.map_draw_timestep_indices = updater_params['map_draw_timestep_indices']
        self.map_draw_freq = updater_params['map_draw_freq']

        

    def update(self, sess, visualize = False):
        batch = {}
        for i, dp in enumerate(self.data_provider):
            provider_batch = dp.dequeue_batch()
            for k in provider_batch:
                if k in batch:
                    batch[k].append(provider_batch[k])
                else:
                    batch[k] = [provider_batch[k]]
        for k in ['action', 'action_post', 'depths1']:
            batch[k] = np.concatenate(batch[k], axis=0)
        feed_dict = {
                self.wm.states : batch[self.state_desc],
                self.wm.action : batch['action'],
                self.wm.action_post : batch['action_post']
            }
        self.targets['global_step'] = self.global_step
        res = sess.run(self.targets, feed_dict = feed_dict)
        global_step = res['global_step']
        if self.map_draw_mode is not None and global_step % self.map_draw_freq == 0:
            if self.map_draw_mode == 'specified_indices':
                map_draw_res = []
                for idx in self.map_draw_example_indices:
                    obs_for_actor = [batch[self.state_desc][idx][t] for t in self.map_draw_timestep_indices]
                    action_samples = self.action_sampler.sample_actions()
                    action, entropy, estimated_world_loss = self.um.act(sess, action_samples, obs_for_actor)
                    to_add = {'example_id' : idx, 'action_sample' : action, 'estimated_world_loss' : estimated_world_loss, 
                            'action_samples' : action_samples, 'depths1' : batch[self.state_desc][idx], 
                            'action' : batch['action'][idx], 'action_post' : batch['action_post'][idx]}
                    map_draw_res.append(to_add)
            res['map_draw'] = map_draw_res
        res = self.postprocessor.postprocess(res, batch)
        return res, global_step





           



class ActionUncertaintyUpdater:
	def __init__(self, models, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params):
                self.data_provider = data_provider \
                        if isinstance(data_provider, list) else [data_provider]
                self.wm = models['world_model']
                self.um = models['uncertainty_model']
                self.postprocessor = postprocessor
                self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
                self.act_lr_params, act_lr = get_learning_rate(self.global_step, ** learning_rate_params['world_model']['act_model'])
                self.um_lr_params, um_lr = get_learning_rate(self.global_step, ** learning_rate_params['uncertainty_model'])
		act_opt_params, act_opt = get_optimizer(act_lr, self.wm.act_loss, self.global_step, optimizer_params['world_model']['act_model'], var_list = self.wm.act_var_list + self.wm.encode_var_list)
                um_opt_params, um_opt = get_optimizer(um_lr, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'], var_list = self.um.var_list)
                self.global_step = self.global_step / 2
	

		self.targets = {'act_pred' : self.wm.act_pred, 'act_loss' : self.wm.act_loss,
				'act_optimizer' : act_opt, 'um_optimizer' : um_opt,
				'estimated_world_loss' : self.um.estimated_world_loss,
				'um_loss' : self.um.uncertainty_loss, 'loss_per_example' : self.um.true_loss,
				'global_step' : self.global_step}

        def update(self, sess, visualize = False):
                batch = {}
                for i, dp in enumerate(self.data_provider):
                    provider_batch = dp.dequeue_batch()
                    for k in provider_batch:
                        if k in batch:
                            batch[k].append(provider_batch[k])
                        else:
                            batch[k] = [provider_batch[k]]
                for k in ['action', 'action_post', 'depths1']:
                    batch[k] = np.concatenate(batch[k], axis=0)
                state_desc = 'depths1'
                #depths, actions, actions_post, next_depth = postprocess_batch_depth(batch, state_desc)
                feed_dict = {
                        self.wm.states : batch[state_desc],
                        self.wm.action : batch['action'],
                        self.wm.action_post : batch['action_post']
                }
                self.targets['global_step'] = self.global_step
                res = sess.run(self.targets, feed_dict = feed_dict)
                glstep = res['global_step']
                res = self.postprocessor.postprocess(res, batch)
                return res, glstep



class LatentUncertaintyUpdater:
	def __init__(self, world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor, updater_params = None):
		self.data_provider = data_provider
		self.wm = world_model
		self.um = uncertainty_model
		self.postprocessor = postprocessor
		self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
		self.act_lr_params, act_lr = get_learning_rate(self.global_step, ** learning_rate_params['world_model']['act_model'])
		self.fut_lr_params, fut_lr = get_learning_rate(self.global_step, ** learning_rate_params['world_model']['fut_model'])
		self.um_lr_params, um_lr = get_learning_rate(self.global_step, ** learning_rate_params['uncertainty_model'])
		act_opt_params, act_opt = get_optimizer(act_lr, self.wm.act_loss, self.global_step, optimizer_params['world_model']['act_model'], var_list = self.wm.act_var_list + self.wm.encode_var_list)
		fut_opt_params, fut_opt = get_optimizer(fut_lr, self.wm.fut_loss, self.global_step, optimizer_params['world_model']['fut_model'], var_list = self.wm.fut_var_list)
		um_opt_params, um_opt = get_optimizer(um_lr, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'], var_list = self.um.var_list)
		self.global_step = self.global_step / 3
		self.targets = {'encoding_i' : self.wm.encoding_i, 'encoding_f' : self.wm.encoding_f,  
						'fut_pred' : self.wm.fut_pred, 'act_pred' : self.wm.act_pred, 
						'act_optimizer' : act_opt, 'fut_optimizer' : fut_opt, 
						'act_lr' : act_lr, 'fut_lr' : fut_lr,
						'fut_loss' : self.wm.fut_loss, 'act_loss' : self.wm.act_loss,
						'estimated_world_loss' : self.um.estimated_world_loss
						}
		self.targets.update({'um_loss' : self.um.uncertainty_loss, 'um_lr' : um_lr, 'um_optimizer' : um_opt, 
						'global_step' : self.global_step, 'loss_per_example' : self.um.true_loss})
		self.state_desc = updater_params['state_desc']
		#checking that we don't have repeat names

	def start(self, sess):
		self.data_provider.start_runner(sess)
		sess.run(tf.global_variables_initializer())

	def update(self, sess, visualize = False):
		batch = self.data_provider.dequeue_batch()
		state_desc = self.state_desc
		#depths, actions, actions_post, next_depth = postprocess_batch_depth(batch, state_desc)
		feed_dict = {
			self.wm.states : batch[state_desc],
			self.wm.action : batch['action'],
			self.wm.action_post : batch['action_post']
		}
		res = sess.run(self.targets, feed_dict = feed_dict)
		res = self.postprocessor.postprocess(res, batch)
		return res


class UncertaintyUpdater:
	def __init__(self, world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params, postprocessor):
		self.data_provider = data_provider
		self.world_model = world_model
		self.um = uncertainty_model
		self.global_step = tf.get_variable('global_step', [], tf.int32, initializer = tf.constant_initializer(0,dtype = tf.int32))
		self.wm_lr_params, wm_learning_rate = get_learning_rate(self.global_step, ** learning_rate_params['world_model'])
		self.wm_opt_params, wm_opt = get_optimizer(wm_learning_rate, self.world_model.loss, self.global_step, optimizer_params['world_model'])
		self.world_model_targets = {'given' : self.world_model.processed_input,  'loss' : self.world_model.loss, 'loss_per_example' : self.world_model.loss_per_example, 'learning_rate' : wm_learning_rate, 'optimizer' : wm_opt, 'prediction' : self.world_model.pred, 'tv' : self.world_model.tv}
		self.inc_step = self.global_step.assign_add(1)
		self.um_lr_params, um_learning_rate = get_learning_rate(self.global_step, **learning_rate_params['uncertainty_model'])
		self.um_lr_params, um_opt = get_optimizer(um_learning_rate, self.um.uncertainty_loss, self.global_step, optimizer_params['uncertainty_model'])
		self.global_step = self.global_step / 2
		self.um_targets = {'loss' : self.um.uncertainty_loss, 'learning_rate' : um_learning_rate, 'optimizer' : um_opt, 'global_step' : self.global_step}
		self.postprocessor = postprocessor
		self.world_action_time = self.world_model.action.get_shape().as_list()[1]
		

	def start(self, sess):
		self.data_provider.start_runner(sess)
		sess.run(tf.global_variables_initializer())

	def update(self, sess, visualize = False):
		batch = self.data_provider.dequeue_batch()
		state_desc = self.um.state_descriptor
		wm_feed_dict = {
			self.world_model.states : batch[state_desc],
			self.world_model.action : batch['action'][:, -self.world_action_time : ]
		}
		world_model_res = sess.run(self.world_model_targets, feed_dict = wm_feed_dict)
		um_feed_dict = {
			self.um.s_i : batch[state_desc][:, :-1],
			self.um.action_sample : batch['action'][:, -1],
			self.um.true_loss : world_model_res['loss_per_example']
		}
		um_res = sess.run(self.um_targets, feed_dict = um_feed_dict)
		wm_res_new = dict(('wm_' + k, v) for k, v in world_model_res.iteritems())
		um_res_new = dict(('um_' + k, v) for k, v in um_res.iteritems())
		wm_res_new.update(um_res_new)
		res = wm_res_new
		res['global_step'] = res.pop('um_global_step')
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
		res['global_step'] = res.pop('um_global_step')
		res = self.postprocessor.postprocess(wm_res_new, batch)
		return res











