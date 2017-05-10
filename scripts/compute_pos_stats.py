


import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import numpy as np
import cPickle
import os
import tensorflow as tf

import tfutils.data as d
import tfutils.base as b

from curiosity.data.short_long_sequence_data import ShortLongSequenceDataProvider
DATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfdata'
VALDATA_PATH = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata'
DATA_BATCH_SIZE = 256
SHORT_LEN = 3
LONG_LEN = 4
MIN_LEN = 4
SAVE_DIR = '/mnt/fs0/nhaber/jerk_stats'
SAVE_LOC = os.path.join(SAVE_DIR, 'all_pulled4.pkl')
JERK_SAVE_LOC = os.path.join(SAVE_DIR, 'jerks4.pkl')



if not os.path.exists(SAVE_DIR):
	os.mkdir(SAVE_DIR)

def table_norot_grab_func(path):
        all_filenames = os.listdir(path)
        print('got to file grabber!')
        return [os.path.join(path, fn) for fn in all_filenames if '.tfrecords' in fn and 'TABLE' in fn and ':ROT:' not in fn]


def pull_all_positions_references():
	dp =  ShortLongSequenceDataProvider(
		data_path = DATA_PATH,
		short_sources = [],
		long_sources = ['object_data', 'reference_ids'],
		short_len = SHORT_LEN,
		long_len = LONG_LEN,
		min_len = MIN_LEN,
		filters = ['is_not_teleporting', 'is_object_there'],
		shuffle = False,
		n_threads = 1,
		batch_size = DATA_BATCH_SIZE,
		file_grab_func = table_norot_grab_func,
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
        inputs = queue.dequeue_many(DATA_BATCH_SIZE)
	to_save = {'pos' : [], 'reference_ids' : [], 'scrn_pos' : []}
	for bn in range(115 * 20):#just tables, no just rot
		print(bn)
		res = sess.run(inputs)
		to_save['pos'].append(res['object_data'][:, :, 0, 5:8])
		to_save['scrn_pos'].append(res['object_data'][:, :, 0, 8:10])
		to_save['reference_ids'].append(res['reference_ids'])
		if bn % 115 == 0:
			with open(SAVE_LOC, 'w') as stream:
				cPickle.dump(to_save, stream)
	with open(SAVE_LOC, 'w') as stream:
		cPickle.dump(to_save, stream)


def load_dat():
	print('loading positions')
	with open(SAVE_LOC) as stream:
		retval = cPickle.load(stream)
	print('positions loaded')
	return retval

def compute_jerks():
	pos_data = load_dat()
	batch_jerk_list = []
	for batch_pos in pos_data['pos']:
		batch_vel = batch_pos[:, 1:] - batch_pos[:,:-1]
		batch_acc = batch_vel[:, 1:] - batch_vel[:,:-1]
		batch_jerk = batch_acc[:,1:] - batch_acc[:,:-1]
		batch_jerk_list.append(batch_jerk)
	return batch_jerk_list

def save_jerks():
	jerks = compute_jerks()
	with open(JERK_SAVE_LOC, 'w') as stream:
		cPickle.dump(jerks, stream)







if __name__ == '__main__':
#	b.get_params()
#	pull_all_positions_references()
	save_jerks()




				
