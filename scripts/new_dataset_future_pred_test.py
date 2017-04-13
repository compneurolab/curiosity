'''
Testing data provider and model base for option Normals +_2 Objects +_2 Actions -> objects (1d vec).
'''

import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import tfutils.data as d
import tfutils.base as b
import tensorflow as tf
import json
from curiosity.data.threeworld_data import ThreeWorldDataProvider
from curiosity.models import explicit_future_prediction_base as fp_base

DATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'

dp_kwargs = {
	'sequence_len' : 4,
	'filters' : ['is_not_teleporting']
}


def test_no_base():
	dp = ThreeWorldDataProvider(DATA_PATH, ['normals', 'actions', 'object_data'], **dp_kwargs)
	sess = tf.Session()
	ops = dp.init_ops()
	queue = b.get_queue(ops[0], queue_type = 'random')
	enqueue_ops = []
	for op in ops:
	    enqueue_ops.append(queue.enqueue_many(op))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
	tf.train.start_queue_runners(sess=sess)
	inputs = queue.dequeue_many(256)
	for i in range(1):
		res = sess.run(inputs)
		print(res.keys())
		print(res['object_data'].shape)
		print(res['actions'].shape)
		print(res['normals'].shape)


if __name__ == '__main__':
	test_no_base()


