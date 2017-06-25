'''
train.py
has the actual loop for training, given all the parameters and things.
This should eventually be like tfutils.base, though for now we specialize a lot.
'''

import curiosity.interaction.environment as environment
import curiosity.interaction.data as data
from curiosity.interaction.data import SimpleSamplingInteractiveDataProvider, SillyLittleListerator
from curiosity.interaction.models import UncertaintyModel, DepthFuturePredictionWorldModel, UniformActionSampler
from curiosity.interaction.update_step import UncertaintyUpdater
import tensorflow as tf

def train(
		optimizer_params = None,
		learning_rate_params = None,
		model_params = None,
		visualize = False,
		data_params = None
	):
	#set up models
	cfg = model_params['cfg']
	world_model = DepthFuturePredictionWorldModel(cfg['world_model'])
	uncertainty_model = UncertaintyModel(cfg['uncertainty_model'])

	#set up data provider
	state_memory_len = {
		'depth' : 2
	}
	rescale_dict = {
		'depth' : (64, 64)
	}
	action_to_message = lambda action, env : environment.normalized_action_to_ego_force_torque(action, env, data_params['action_limits'])
	env = environment.Environment(1, 1, action_to_message, state_memory_len = state_memory_len, rescale_dict = rescale_dict)
	scene_infos = data.SillyLittleListerator([environment.example_scene_info])
	steps_per_scene = data.SillyLittleListerator([float('inf')])
	data_provider = SimpleSamplingInteractiveDataProvider(env, uncertainty_model, 1, scene_infos, steps_per_scene, UniformActionSampler(cfg), capacity = 5)

	#set up updater
	updater = UncertaintyUpdater(world_model, uncertainty_model, data_provider, optimizer_params, learning_rate_params)

	#do the training loop!
	sess = tf.Session()
	updater.start(sess)
	while True:
		updater.update(sess, visualize)