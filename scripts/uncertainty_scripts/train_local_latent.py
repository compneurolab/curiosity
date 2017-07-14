'''
Local test for latent space model training.
'''


import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train
from curiosity.interaction.models import mario_world_model_config
from tfutils import base, optimizer
import numpy as np

NUM_BATCHES_PER_EPOCH = 1e8

STATE_DESC = 'depths1'


params = {
	'model_params' : {
		'cfg' : {
				'world_model' : mario_world_model_config,
				'uncertainty_model' : {
					'state_shape' : [2, 128, 170, 3],
					'action_dim' : 8,
					'n_action_samples' : 50,
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
					'state_descriptor' : STATE_DESC
				},
				'seed' : 0
			},
	},

	'optimizer_params' : {
		'world_model' : {
			'fut_model' : {
				'func': optimizer.ClipOptimizer,
				'optimizer_class': tf.train.AdamOptimizer,
				'clip': True,
			},
			'act_model' : {
				'func': optimizer.ClipOptimizer,
				'optimizer_class': tf.train.AdamOptimizer,
				'clip': True,
			}
		},
		'uncertainty_model' : {
			'func': optimizer.ClipOptimizer,
			'optimizer_class': tf.train.AdamOptimizer,
			'clip': True,
		}

	},


	'learning_rate_params' : {
		'world_model' : {
			'act_model' : {
				'func': tf.train.exponential_decay,
				'learning_rate': 1e-5,
				'decay_rate': 1.,
				'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
				'staircase': True
			},
			'fut_model' : {
				'func': tf.train.exponential_decay,
				'learning_rate': 1e-5,
				'decay_rate': 1.,
				'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
				'staircase': True
			}
		},
		'uncertainty_model' : {
			'func': tf.train.exponential_decay,
			'learning_rate': 1e-5,
			'decay_rate': 1.,
			'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
			'staircase': True
		}
	},


	'data_params' : {
		'action_limits' : np.array([1., 1.] + [80. for _ in range(6)]),
		'full_info_action' : True
	},

	'visualize' : True,

	'exp_id' : 'test_latent'

}


if __name__ == '__main__':
	train.train_local(**params)

