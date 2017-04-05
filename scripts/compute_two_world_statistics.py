import tensorflow as tf
import h5py
import make_new_tfrecord as mnt
import cPickle
import os
import sys
import numpy as np

BATCH_SIZE = mnt.BATCH_SIZE
IS_IMAGE = [True, True, True, True, True, True] + [False for _ in range(len(mnt.ATTRIBUTE_NAMES) - 6)]
IS_IMAGE = dict(x for x in zip(mnt.ATTRIBUTE_NAMES, IS_IMAGE))
SAVE_DIR = '/media/data2/two_world_dataset/statistics'


def online_mean(X_so_far, X_new, num_seen_so_far):
	num_seen_so_far = tf.cast(num_seen_so_far, tf.float32)
	return num_seen_so_far / (num_seen_so_far + 1) * X_so_far + X_new / (num_seen_so_far + 1)

def add_average_online(X_so_far, X_batch_new, num_seen_so_far):
	X_new = tf.reduce_mean(X_batch_new, axis = 0)
	return online_mean(X_so_far, X_new, num_seen_so_far)

def update_statistics(mean_so_far, square_so_far, min_so_far, max_so_far, batch_attribute_data, num_seen_so_far):
	square_res = add_average_online(square_so_far, tf.square(batch_attribute_data), num_seen_so_far)
	mean_res = add_average_online(mean_so_far, batch_attribute_data, num_seen_so_far)
	max_res = tf.maximum(max_so_far, tf.reduce_max(batch_attribute_data, axis = 0))
	min_res = tf.maximum(min_so_far, tf.reduce_min(batch_attribute_data, axis = 0))
	return [mean_res, square_res, min_res, max_res]


my_batch_data = dict((k, tf.placeholder(tf.uint8 if IS_IMAGE[k] else tf.float32, [BATCH_SIZE] + list(shp))) for k, shp in mnt.ATTRIBUTE_SHAPES.iteritems())
current_results = dict((k, [tf.placeholder(tf.float32, shp) for _ in range(4)]) for k, shp in mnt.ATTRIBUTE_SHAPES.iteritems())

initial_results = dict((k, [np.zeros(shp, dtype = np.uint8 if IS_IMAGE[k] else np.float32) for _ in range(4)]) for k, shp in mnt.ATTRIBUTE_SHAPES.iteritems())
current_count = tf.placeholder(tf.int32)

def update_batch_data(so_far_dict, batch_data, num_seen_so_far):
	retval = {}
	for k in batch_data:
		mean_so_far, square_so_far, min_so_far, max_so_far = so_far_dict[k]
		batch_data_converted = tf.cast(batch_data[k], tf.float32)
		retval[k] = update_statistics(mean_so_far, square_so_far, min_so_far, max_so_far, batch_data_converted, num_seen_so_far)
	return retval

updated_results = update_batch_data(current_results, my_batch_data, current_count)

def do_calculation(job_num):
	if not os.path.exists(SAVE_DIR):
		os.mkdir(SAVE_DIR)
	results_so_far = initial_results
	with open(mnt.JOBS_DIVISION) as stream:
		buckets, types_dict = cPickle.load(stream)
	my_bucket = buckets[job_num]
	num_seen_so_far = 0
	with tf.Session() as sess:
		for batch_type in my_bucket:
			print('Stats for type ' + batch_type)
			for bn in types_dict[batch_type]:
				print('batch num ' + str(bn))
				print('num seen so far: ' + str(num_seen_so_far))
				batch_data_dict = mnt.get_batch_data(bn)
				my_feed_dict = dict((my_batch_data[k],batch_data_dict[k]) for k in my_batch_data)
				my_feed_dict.update(dict((current_results[k][i], results_so_far[k][i]) for k in current_results for i in range(4)))
				my_feed_dict.update({current_count : num_seen_so_far})
				results_so_far = sess.run(updated_results, feed_dict = my_feed_dict)
				num_seen_so_far += 1
				with open(os.path.join(SAVE_DIR, 'partition_' + str(job_num) + '.p'), 'w') as stream:
					cPickle.dump((results_so_far, num_seen_so_far), stream)


if __name__ == '__main__':
	job_num_seed = int(sys.argv[2])
	job_nums = [job_num_seed + i for i in range(4)]
	os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
	for job_num in job_nums:
		do_calculation(job_num)



