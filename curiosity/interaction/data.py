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
        self.next_state = None

    def add(self, **kwargs):
    	self.states.append(kwargs['state'])
    	self.actions.append(kwargs['action'])
    	self.next_state = kwargs['next_state']


class SimpleSamplingInteractiveDataProvider(threading.Thread):
	def __init__(self, environment, policy, batch_size, initializations, num_steps_per_scene, action_sampler, capacity = 5):
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

	def start_runner(self, sess):
		self.sess = sess
		self.start()

	def run(self):
		with self.sess.as_default():
			self._run()

	def run_env(self):
		obs = self.env.next_config(* self.scene_params.next())
		num_this_scene = 0
		scene_len = self.scene_lengths.next()

		while True:
			recent_history = SimpleRecentHistory()
			for _ in range(self.batch_size):
				if num_this_scene >= scene_len:
					obs = self.env.next_config(* self.scene_params.next())
					num_this_scene = 0
					scene_len = self.scene_lengths.next()
					break
				action_sample = self.action_sampler.sample_actions()
				action = self.policy.act(self.sess, action_sample, np.array([obs['depth']]))

				new_obs = self.env.step(action)
				recent_history.add(state = obs, next_state = new_obs, 
								action = action)
				obs = new_obs


			yield recent_history

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

