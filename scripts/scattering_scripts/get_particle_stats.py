'''
Makes basic stats file for our fun new dataset. This time, reads from tfrecord writers.
'''
import tensorflow as tf
import sys
sys.path.append('tfutils')
sys.path.append('curiosity')
import tfutils.data as d
import tfutils.base as b
from curiosity.data.short_long_sequence_data import ShortLongSequenceDataProvider
import numpy as np
import cPickle 
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
DATA_PATH = '/mnt/fs1/datasets/eight_world_dataset/tfdata'
NUM_BATCHES = 6 * 64
BATCH_SIZE = 256.
ATTRIBUTES = ['actions', 'agent_data', 'depths', 'images', 'is_acting', \
        'is_not_dropping', 'is_not_teleporting', 'is_not_waiting', 'is_object_in_view', \
        'is_object_there', 'max_coordinates', 'min_coordinates', 'object_data', \
        'objects', 'particles', 'reference_ids', 'vels', 'vels_curr']

STATS_SAVE_LOC = '/mnt/fs1/datasets/eight_world_dataset/new_stats/stats.pkl'
STATS_FIXED_SAVE_LOC = '/mnt/fs1/datasets/eight_world_dataset/new_stats/stats_std.pkl'


def get_data_source():
    dp = ShortLongSequenceDataProvider(
            data_path = DATA_PATH,
            short_sources = [],
            long_sources = ATTRIBUTES,
            n_threads = 1,
            min_len = 1,
            short_len = 1,
            long_len = 1)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    ops = dp.init_ops()
    queue = b.get_queue(ops[0], queue_type = 'fifo')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))
    tf.train.queue_runner.add_queue_runner(\
            tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    tf.train.start_queue_runners(sess=sess)
    inputs = queue.dequeue_many(int(BATCH_SIZE))
    for k in inputs:
        inputs[k] = inputs[k][:, 0]
        if k in ['particles']:
            inputs[k] = tf.reshape(inputs[k], [int(BATCH_SIZE), -1, 7])
    return sess, inputs


def compute_batch_var(batch_data, batch_mean, key):
    delta = batch_data - batch_mean
    return tf.reduce_sum(delta**2, axis = 0) / float(BATCH_SIZE - 1)


def update_mean_var(old_mean, old_var, num_batches_seen, batch_stats):
    batch_mean = batch_stats['mean']
    batch_var = batch_stats['var']
    delta = batch_mean - old_mean
    m_batch = batch_var * (BATCH_SIZE - 1)
    m_old = old_var * (num_batches_seen * BATCH_SIZE - 1)
    M2 = m_batch +  m_old + (delta**2) * BATCH_SIZE \
            * num_batches_seen / (num_batches_seen + 1)
    new_mean = old_mean + delta / num_batches_seen
    return new_mean, M2 / ((num_batches_seen + 1) * BATCH_SIZE - 1)


def update_stats(old_stats, batch_stats, num_batches_seen):
    mean = old_stats['mean']
    var = old_stats['var']
    mean, var = update_mean_var(mean, var, num_batches_seen, batch_stats)
    max_res = np.maximum(old_stats['max'], batch_stats['max'])
    min_res = np.minimum(old_stats['min'], batch_stats['min'])
    return {'max' : max_res, 'min' : min_res, 'mean' : mean, 'var' : var}


def get_batch_stats(inputs, key):
    batch_data = inputs[key]
    batch_data = tf.cast(batch_data, tf.float32)
    if key in ['actions']:
        batch_mean = tf.reduce_mean(batch_data, axis = [0,1])
        batch_mean = tf.tile(tf.expand_dims(batch_mean, axis=0), \
                [tf.shape(batch_data)[1],1])
    elif key in ['particles']:
        batch_mean = tf.reduce_sum(batch_data, axis = [0,1])
        n = tf.reduce_sum(inputs['object_data'][:,:,13] / 7, axis = [0,1])
        batch_mean /= n
        batch_mean = tf.tile(tf.expand_dims(batch_mean, axis=0), \
                [tf.shape(batch_data)[1],1])
    else:
        batch_mean = tf.reduce_mean(batch_data, axis = 0)

    batch_var = compute_batch_var(batch_data, batch_mean, key)
    if key in ['actions']:
        my_max = tf.reduce_max(batch_data, axis = [0,1])
        my_min = tf.reduce_min(batch_data, axis = [0,1])
        my_max = tf.tile(tf.expand_dims(my_max, axis=0), \
                [tf.shape(batch_data)[1],1])
        my_min = tf.tile(tf.expand_dims(my_min, axis=0), \
                [tf.shape(batch_data)[1],1])
    elif key in ['particles']:
        # process each dimension separately
        batch_data = tf.unstack(batch_data, axis=2)
        my_maxs = []
        my_mins = []
        for data in batch_data:
            # remove zero values
            non_zero_data = tf.boolean_mask(data, tf.not_equal(data, 0))
            my_maxs.append(tf.reduce_max(non_zero_data))
            my_mins.append(tf.reduce_min(non_zero_data))
        my_max = tf.stack(my_maxs, axis=0)
        my_min = tf.stack(my_mins, axis=0)
        batch_data = tf.stack(batch_data, axis=2)
        my_max = tf.tile(tf.expand_dims(my_max, axis=0), \
                [tf.shape(batch_data)[1],1])
        my_min = tf.tile(tf.expand_dims(my_min, axis=0), \
                [tf.shape(batch_data)[1],1])
    else:
        my_max = tf.reduce_max(batch_data, axis = 0)
        my_min = tf.reduce_min(batch_data, axis = 0)
    return {'max' : my_max, 'min' : my_min, 'mean' : batch_mean, 'var' : batch_var}


def get_stats():
    sess, inputs = get_data_source()
    stats = dict((k, get_batch_stats(inputs, k)) for k in inputs)
    for bn in range(NUM_BATCHES):
        print(bn)
        print('retrieving stats')
        batch_stats = sess.run(stats)
        print('stats retrieved')
        if bn == 0:
            stats_so_far = batch_stats
        else:
            for k in stats_so_far:
                stats_so_far[k] = update_stats(stats_so_far[k], \
                        batch_stats[k], float(bn))
        if bn % 10 == 0:
            print('about to write')
            with open(STATS_SAVE_LOC, 'w') as stream:
                cPickle.dump(stats_so_far, stream)
            print('written')
    with open(STATS_SAVE_LOC, 'w') as stream:
        cPickle.dump(stats_so_far, stream)


def whoops_gotta_square_root_std():
    with open(STATS_SAVE_LOC) as stream:
        stats = cPickle.load(stream)
    for k in stats:
        stats[k]['std'] = np.sqrt(stats[k]['var'])
    with open(STATS_FIXED_SAVE_LOC, 'w') as stream:
        cPickle.dump(stats, stream)


if __name__ == '__main__':
    get_stats()
    whoops_gotta_square_root_std()
