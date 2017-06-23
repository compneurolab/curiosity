'''
A local test for uncertain_curiosity loop.
'''

import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf

from curiosity.interaction import train
from curiosity.interaction.models import another_sample_cfg
from tfutils import base, optimizer
import numpy as np

NUM_BATCHES_PER_EPOCH = 1e8

params = {
	'model_params' : {
		'cfg' : another_sample_cfg
	},

	'optimizer_params' : {
		'world_model' : {
			'func': optimizer.ClipOptimizer,
			'optimizer_class': tf.train.AdamOptimizer,
			'clip': True,
		},
		'uncertainty_model' : {
			'func': optimizer.ClipOptimizer,
			'optimizer_class': tf.train.AdamOptimizer,
			'clip': True,
		}

	},


	'learning_rate_params' : {
		'world_model' : {
			'func': tf.train.exponential_decay,
			'learning_rate': 1e-5,
			'decay_rate': 1.,
			'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
			'staircase': True
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
		'action_limits' : np.array([1., 1.] + [80. for _ in range(6)])
	},

	'visualize' : True,

	'exp_id' : 'run15'

}

if __name__ == '__main__':
	train.train(**params)










