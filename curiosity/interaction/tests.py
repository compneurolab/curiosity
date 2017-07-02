'''
Some interactive tests.
'''

import sys
sys.path.append('curiosity')


from curiosity.interaction import environment
import data
import tensorflow as tf
import numpy as np


scene_infos = data.SillyLittleListerator([environment.example_scene_info])
steps_per_scene = data.SillyLittleListerator([150])

env = environment.Environment(1, 1, environment.test_action_to_message_fn, USE_TDW = False, host_address = '10.102.2.162')

class TestPolicy:
	def __init__(self):
		self.counter = 0

	def get_initial_features(self):
		return np.zeros(10), np.zeros(20)

	def act(self, sess, obs, features):
		action = (self.counter // 50) % 3
		action_one_hot = [0, 0, 0]
		action_one_hot[action] = 1
		self.counter += 1
		return np.array(action_one_hot), 0., self.get_initial_features()

def env_walkabout_test():
	scene_info = scene_infos.next()
	obs = env.next_config(* scene_info)
	print obs.keys()
	for act in [0, 1, 2]:
		for _ in range(50):
			obs = env.step(act)
			print obs.keys()


def runner_test():
	runner = data.InteractiveDataProvider(env, TestPolicy(), 10, scene_infos, steps_per_scene)
	sess = tf.Session()
	runner.start_runner(sess)
	for t in range(30):
		batch = runner.dequeue_batch()
		print len(batch.states)

def another_runner_test():
	runner = data.SimpleSamplingInteractiveDataProvider(env, TestPolicy(), 10, scene_infos, steps_per_scene)


if __name__ == '__main__':
	#runner_test()
	env_walkabout_test()

