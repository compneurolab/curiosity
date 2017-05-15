


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
DATA_PATH = '/mnt/fs0/datasets/three_world_dataset/new_tfdata_newobj'
VALDATA_PATH = '/mnt/fs0/datasets/three_world_dataset/new_tfvaldata_newobj'
DATA_BATCH_SIZE = 256
SHORT_LEN = 3
LONG_LEN = 4
MIN_LEN = 4
SAVE_DIR = '/mnt/fs0/nhaber/pos_dat_scattering'
SAVE_LOC = os.path.join(SAVE_DIR, 'tr_pulled.pkl')
VAL_SAVE_LOC = os.path.join(SAVE_DIR, 'val_pulled.pkl')


if not os.path.exists(SAVE_DIR):
	os.mkdir(SAVE_DIR)

def pull_all_positions_references(val = False):
	dat_input_path = DATA_PATH
	save_loc = SAVE_LOC
	if val:
		dat_input_path = VALDATA_PATH
		save_loc = VAL_SAVE_LOC
	dp =  ShortLongSequenceDataProvider(
		data_path = dat_input_path,
		short_sources = [],
		long_sources = ['object_data', 'reference_ids'],
		short_len = SHORT_LEN,
		long_len = LONG_LEN,
		min_len = MIN_LEN,
		filters = ['is_not_teleporting'],
		shuffle = False,
		n_threads = 1,
		batch_size = DATA_BATCH_SIZE,
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
	to_save = {'object_data' : [], 'reference_ids' : []}
	for bn in range(4000 - 6):#just tables, no just rot
		print(bn)
		res = sess.run(inputs)
		to_save['object_data'].append(res['object_data'][:, :, 0])
		to_save['reference_ids'].append(res['reference_ids'])
		if bn % 100 == 0:
			with open(save_loc, 'w') as stream:
				cPickle.dump(to_save, stream)
	with open(save_loc, 'w') as stream:
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
	b.get_params()
	pull_all_positions_references(val = False)
	pull_all_positions_references(val = True)
#	save_jerks()




				
