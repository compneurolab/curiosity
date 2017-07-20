'''
Interactive data providers for agents.
For now, going to use feed dicts until we see a real difference in performance, this seems to be so much easier to work through.
'''

import six.moves.queue as queue
import threading
import numpy as np

class SillyLittleListerator:
	def __init__(self, in_list):
		self.my_list = in_list
		self.next_loc = 0

	def next(self):
		retval = self.my_list[self.next_loc]
		self.next_loc += 1
		self.next_loc = self.next_loc % len(self.my_list)
		return retval

class RecentHistory(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.internal_states = []
        self.next_state = None

    def add(self, **kwargs):
    	self.states.append(kwargs['state'])
    	self.actions.append(kwargs['action'])
    	self.values.append(kwargs['value'])
    	self.internal_states.append(kwargs['internal_state'])
    	self.next_state = kwargs['next_state']

class SimpleRecentHistory(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.messages = []
        self.next_state = None

    def add(self, **kwargs):
    	self.states.append(kwargs['state'])
    	self.actions.append(kwargs['action'])
    	self.messages.append(kwargs['message'])
    	self.next_state = kwargs['next_state']


def batch_FIFO(history, batch_size = 32, data_lengths = {'obs' : {'depths1' : 3}, 'action' : 2, 'action_post' : 2):
	assert len(data_length['obs']) == 1
	batch = {}
	for k, v in data_lengths.iteritems():
		if k == 'obs':
			desc = data_lengths['obs'].keys()[0]
			dat_len = v[desc]
			dat_raw = history['obs'][desc]
		else:
			desc = k
			dat_len = v
			dat_raw = history[k]
		nones_replaced = replace_the_nones(dat_raw[-(batch_size + dat_len - 1) : ])
		batch[desc] = np.array([nones_replaced[sample_num : sample_num + dat_len] for sample_num in range(batch_size)])
	batch['msg'] = history['msg'][-1]
	batch['other'] = history['other'][-1]
	return batch


class BSInteractiveDataProvider(threading.Thread):
	'''
	A batching, sampling interactive data provider.
	Meant to support a light amount of experience replay, as well as simply giving batches of data.
	'''
	def __init__(self, environment, policy, scene_params, scene_lengths, action_sampler, batching_fn, capacity = 5, batch_size = 32, gather_per_batch = 32, gather_at_beginning = 32):
		self.env = environment
		self.policy = policy
		self.batch_size = batch_size
		self.capacity = capacity
		self.queue = queue.Queue(capacity)
		self.daemon = True
		self.sess = None
		self.scene_params = scene_params
		self.scene_lengths = scene_lengths
		self.action_sampler = action_sampler
		self.gather_per_batch = gather_per_batch
		self.gather_at_beginning = gather_at_beginning
		self.batching_fn = batching_fn

	def start_runner(self, sess):
		self.sess = sess
		self.start()

	def run(self):
		with self.sess.as_default():
			self._run()

        def _run(self):
                yielded = self.run_env()
                while True:
			history = next(yielded)
			batch = self.batching_fn(history)
                        self.queue.put(batch, timeout = 600.0)

        def dequeue_batch(self):
                return self.queue.get(timeout = 600.0)

	def run_env(self):
		#initialize counters
		num_this_scene = 0
		scene_len = -1
		total_gathered = 0

		while True:
			#gather a batch
			num_this_yield = 0
			while num_this_yield < self.gather_per_batch or total_gathered < self.gather_at_beginning:
				#check for scene start condition
				if num_this_scene >= scene_len:
					obs, msg = self.env.next_config(* self.scene_params.next())
					num_this_scene = 0
					scene_len = self.scene_lengths.next()
					action = None
					for _ in range(
				#select action and act on world
				action_sample = self.action_sampler.sample_actions()
				action, entropy, estimated_world_loss = self.policy.act(self.sess, action_sample, obs, full_info = True)
				action = self.policy.act(self.sess, action_sample, obs)
				obs, msg, action, action_post, other_mem = self.env.step(action, other_data = (entropy, estimated_world_loss, action_sample))
			

				#update counters
				num_this_yield += 1
				total_gathered += 1
				if action is not None:
					yield {'observation' : obs,'msg' : msg, 'action' : action, 'action_post' : action_post, 'other' : other_mem}
	
		


class SimpleSamplingInteractiveDataProvider(threading.Thread):
	def __init__(self, environment, policy, batch_size, initializations, num_steps_per_scene, action_sampler, full_info_action = False, capacity = 5):
		threading.Thread.__init__(self)
		self.policy = policy
		self.batch_size = batch_size
		self.env = environment
		self.capacity = capacity
		self.queue = queue.Queue(capacity)
		self.daemon = True
		self.sess = None
		self.scene_params = initializations
		self.scene_lengths = num_steps_per_scene
		self.action_sampler = action_sampler
		self.full_info_action = full_info_action

	def start_runner(self, sess):
		self.sess = sess
		self.start()

	def run(self):
		with self.sess.as_default():
			self._run()

	def run_env(self):
		obs, msg = self.env.next_config(* self.scene_params.next())
		num_this_scene = 0
		scene_len = self.scene_lengths.next()
		action = None

		while True:
			if num_this_scene >= scene_len:
				obs, msg = self.env.next_config(* self.scene_params.next())
				num_this_scene = 0
				scene_len = self.scene_lengths.next()
				action = None

			action_sample = self.action_sampler.sample_actions()
			if self.full_info_action:
				action, entropy, estimated_world_loss = self.policy.act(self.sess, action_sample, obs, full_info = True)
			else:
				action = self.policy.act(self.sess, action_sample, obs)
			obs, msg, action, action_post = self.env.step(action)
			if self.full_info_action:
				assert 'entropy' not in obs and 'est_loss' not in obs
				obs['entropy'] = entropy
				obs['est_loss'] = estimated_world_loss
				obs['action_sample'] = action_sample
			num_this_scene += 1
			if action is not None:
				yield obs, msg, action, action_post

	def _run(self):
		yielded = self.run_env()
		while True:
			self.queue.put(next(yielded), timeout = 600.0)

	def dequeue_batch(self):
		return self.queue.get(timeout = 600.0)


class InteractiveDataProvider(threading.Thread):
	def __init__(self,
			environment,
			policy,
			batch_size,
			initializations,
			num_steps_per_scene,
			capacity = 5,
		):
		threading.Thread.__init__(self)
		self.policy = policy
		self.batch_size = batch_size
		self.env = environment
		self.capacity = capacity
		self.queue = queue.Queue(capacity)
		self.last_features = None
		self.daemon = True
		self.sess = None
		self.scene_params = initializations
		self.scene_lengths = num_steps_per_scene

	def start_runner(self, sess):
		self.sess = sess
		self.start()

	def run(self):
		with self.sess.as_default():
			self._run()

	def run_env(self):
		obs = self.env.next_config(* self.scene_params.next())
		features = self.policy.get_initial_features()
		num_this_scene = 0
		scene_len = self.scene_lengths.next()

		while True:
			recent_history = RecentHistory()
			for _ in range(self.batch_size):
				if num_this_scene >= scene_len:
					obs = self.env.next_config(* self.scene_params.next())
					features = self.policy.get_initial_features()
					num_this_scene = 0
					scene_len = self.scene_lengths.next()
					break
				action, value, new_features = self.policy.act(self.sess, obs, features)
				new_obs = self.env.step(action.argmax())
				recent_history.add(state = obs, next_state = new_obs, 
								action = action, value = value, internal_state = features)
				obs = new_obs
				features = new_features


			yield recent_history

	def _run(self):
		yielded = self.run_env()
		while True:
			self.queue.put(next(yielded), timeout = 600.0)

	def dequeue_batch(self):
		return self.queue.get(timeout = 600.0)

