'''
Meta-making script for new dataset
'''

import tensorflow as tf
import os
import cPickle

# DATA_LOC = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
# VALIDATION_DATA_LOC = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata'
# ATTRIBUTE = 'object_data2'
# DTYPE_STRING = True
# RAW_TYPE = tf.float32
# RAW_SHAPE = [11, 12]
# D_TYPE = tf.uint8
# SHAPE = [160, 375, 3]

DATA_LOC = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
VALIDATION_DATA_LOC = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata'
ATTRIBUTE = 'reference_ids'
DTYPE_STRING = True
RAW_TYPE = tf.int32
RAW_SHAPE = [2,]
# D_TYPE = tf.uint8
# SHAPE = [160, 375, 3]


to_write = {}
if DTYPE_STRING:
	to_write['dtype'] = tf.string
	to_write['shape'] = []
else:
	to_write['dtype'] = D_TYPE
	to_write['shape'] = SHAPE
if RAW_TYPE is not None:
	to_write['rawtype'] = RAW_TYPE
	to_write['rawshape'] = RAW_SHAPE

print('writing!')
to_write = {ATTRIBUTE : to_write}
print(to_write)

with open(os.path.join(DATA_LOC, ATTRIBUTE, 'meta.pkl'), 'w') as stream:
	cPickle.dump(to_write, stream)

with open(os.path.join(VALIDATION_DATA_LOC, ATTRIBUTE, 'meta.pkl'), 'w') as stream:
	cPickle.dump(to_write, stream)