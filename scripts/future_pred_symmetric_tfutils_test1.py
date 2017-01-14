import os
import copy

import numpy as np
import tensorflow as tf
import sys

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



# cfg0 = postprocess_config(json.load(open(cfgfile)))
cfg0 = {}
seed = 0
rng = np.random.RandomState(seed=seed)


batch_size = 64
IMAGE_SIZE = 512
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


outputs, params = modelsource.model_tfutils(inputs, rng, cfg_initial = cfg0, train = True, slippage = 0)