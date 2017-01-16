import os
import copy

import numpy as np
import tensorflow as tf
import sys
import json

sys.path.append('curiosity')

# import curiosity.utils.base as base
from curiosity.models import future_pred_symmetric_coupled_with_below
import curiosity.models.future_pred_symmetric_coupled_with_below as modelsource
# import curiosity.datasources.images_futures_and_actions as datasource
CODE_ROOT = os.environ['CODE_BASE']
cfgfile = os.path.join(CODE_ROOT, 
                       'curiosity/curiosity/configs/future_test_config_b.cfg')


from curiosity.utils.loadsave import (get_checkpoint_path,
                                      preprocess_config,
                                      postprocess_config)



cfg0 = postprocess_config(json.load(open(cfgfile)))
# print cfg0
# cfg0 = {}
seed = 0
rng = np.random.RandomState(seed=seed)


batch_size = 64
IMAGE_SIZE = 360
NUM_CHANNELS = 3
ACTION_LENGTH = 10
current_node = tf.placeholder(
  tf.float32,
  shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

future_node = tf.placeholder(
    tf.float32,
  shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

actions_node = tf.placeholder(tf.float32,
                            shape=(batch_size,
                                   ACTION_LENGTH))

time_node = tf.placeholder(tf.float32,
                         shape=(batch_size, 1))

inputs = {'current' : current_node, 'future' : future_node, 'action' : actions_node, 'time' : time_node}


outputs, params = modelsource.model_tfutils(inputs, rng, cfg = cfg0, train = True, slippage = .5)

all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

for var in all_vars:
	print var.name

real_encode_depth = len([k for k in params.keys() if 'enc' in k])

for i in range(real_encode_depth + 1):
	print(outputs['pred']['pred' + str(i)])
	print(outputs['future']['future' + str(i)])
