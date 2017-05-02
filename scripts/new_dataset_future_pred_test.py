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
from curiosity.data.short_long_sequence_data import ShortLongSequenceDataProvider
from curiosity.models import explicit_future_prediction_base as fp_base
import time
import os
from PIL import Image
import numpy as np

DATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
TIME_SEEN = 3
WRITE_DIR = '/home/nhaber/temp/data_provider_test'

dp_kwargs = {
	'sequence_len' : 10,
	'filters' : ['is_not_teleporting']
}


def test_no_base():
	dp = ThreeWorldDataProvider(DATA_PATH, ['normals', 'normals2', 'actions', 'object_data'], **dp_kwargs)
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False))
	ops = dp.init_ops()
	queue = b.get_queue(ops[0], queue_type = 'random')
	enqueue_ops = []
	for op in ops:
	    enqueue_ops.append(queue.enqueue_many(op))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
	tf.train.start_queue_runners(sess=sess)
	inputs = queue.dequeue_many(256)
	for i in range(5):
		start = time.time()
		res = sess.run(inputs)
		print(time.time() - start)
		print(res.keys())
		print(res['object_data'].shape)
		print(res['actions'].shape)
		print(res['normals'].shape)

def test_long_sequence_no_base():
	dp = ShortLongSequenceDataProvider(DATA_PATH,
			short_sources = ['normals', 'normals2'],
			long_sources = ['actions', 'object_data', 'reference_ids'],
			short_len = 3,
			long_len = 23,
			min_len = 6,
			filters = ['is_not_teleporting', 'is_object_there'],
			is_there_subsetting_rule = 'just_first'
		)
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False))
	ops = dp.init_ops()
	queue = b.get_queue(ops[0], queue_type = 'fifo')
	enqueue_ops = []
	for op in ops:
	    enqueue_ops.append(queue.enqueue_many(op))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
	tf.train.start_queue_runners(sess=sess)
	inputs = queue.dequeue_many(256)
	res_collector = []
	for i in range(1):
		start = time.time()
		res = sess.run(inputs)
		res_collector.append(res)
		print(time.time() - start)
#		print(res.keys())
#		print(res['object_data'].shape)
#		print(res['reference_ids'].shape)
#		print(res['normals'].shape)
#		print(res['normals2'].shape)
#		print(res['reference_ids'][0])
#		print(res['reference_ids'][255])
#		for i in range(256):
#			print(i)
#			print(res['master_filter'][i])
	return res_collector

def convert_for_write(arr):
	# my_max = np.amax(arr)
	# my_min = np.amin(arr)
	# if my_min - my_max < .01:
	# 	my_max = my_min + 1
	arr = 255. * arr
	return arr.astype(np.uint8)


def write_results(res, res_num, range_to_draw):
	if not os.path.exists(WRITE_DIR):
		os.mkdir(WRITE_DIR)
	for ex_num in range_to_draw:
		ex_dir = os.path.join(WRITE_DIR, 'ex_' + str(ex_num))
		if not os.path.exists(ex_dir):
			os.mkdir(ex_dir)
		for im_type in ['normals', 'normals2', 'object_data_seen', 'actions_seen', 'actions_future']:
			type_write_dir = os.path.join(ex_dir, im_type)
			if not os.path.exists(type_write_dir):
				os.mkdir(type_write_dir)
			dat = res[im_type][ex_num]
			for t in range(dat.shape[0]):
				my_dat = dat[t]
				if im_type == 'normals' or im_type == 'normals2':
					fn = os.path.join(type_write_dir, str(t) + '.png')
					my_dat = my_dat.astype(np.uint8)
					im = Image.fromarray(my_dat)
					im.save(fn)
				elif im_type == 'object_data_seen':
					my_dat = convert_for_write(my_dat)
					fn1 = os.path.join(type_write_dir, str(t) + '_obj1.png')
					fn2 = os.path.join(type_write_dir, str(t) + '_obj2.png')
					im1 = Image.fromarray(my_dat[:, :, 0:3])
					im2 = Image.fromarray(my_dat[:, :, 4:7])
					im1.save(fn1)
					im2.save(fn2)
				elif im_type == 'actions_seen' or im_type == 'actions_future':
					my_dat = convert_for_write(my_dat)
					fn1 = os.path.join(type_write_dir, str(t) + '_force.png')
					fn2 = os.path.join(type_write_dir, str(t) + '_tor.png')
					im1 = Image.fromarray(my_dat[:, :, 0:3])
					im2 = Image.fromarray(my_dat[:, :, 3:])
					im1.save(fn1)
					im2.save(fn2)




def test_base():
	dp = ThreeWorldDataProvider(DATA_PATH, ['normals', 'normals2', 'actions', 'object_data'], **dp_kwargs)
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False))
	ops = dp.init_ops()
	queue = b.get_queue(ops[0], queue_type = 'random')
	enqueue_ops = []
	for op in ops:
	    enqueue_ops.append(queue.enqueue_many(op))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
	tf.train.start_queue_runners(sess=sess)
	inputs = queue.dequeue_many(32)
	net = fp_base.FuturePredictionBaseModel(inputs, TIME_SEEN)
	for i in range(5):
		start = time.time()
		res = sess.run(net.inputs)
		print(res.keys())
		print(time.time() - start)
		print(res['normals'].shape)
		print(res['object_data_seen'].shape)
		print(res['actions_seen'].shape)
		print(res['actions_future'].shape)
		print(res['object_data_future'].shape)
		if i == 0:
			write_results(res, i, range(20,25))



if __name__ == '__main__':
	b.get_params()
	stuff = test_long_sequence_no_base()

