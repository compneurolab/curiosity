'''
Playing with data providers for explicit position problem.
'''

import sys
sys.path.append('tfutils')
sys.path.append('curiosity')




import numpy as np
from numpy.testing import assert_allclose, assert_equal
import os
import tfutils.data as d
import tfutils.base as b
import tensorflow as tf
import json
from curiosity.data.explicit_positions import PositionPredictionData

os.environ['CUDA_VISIBLE_DEVICES'] = '3'



DATA_PATH = '/media/data2/one_world_dataset/tfdata'

source_paths = [os.path.join(DATA_PATH, 'worldinfo')]

def see_first_key(json_np_arrs):
	ret_list = []
	my_batch = json_np_arrs[0]
	for frame_ex in my_batch:
		json_res = json.loads(frame_ex)
		ret_list.append([json_res.keys()[0]])
	retval = np.array(ret_list).astype('string')
	return retval

def screen_out_statics(observed_objects):
    return [obj for obj in observed_objects if (not obj[-1]) or obj[1] == -1]

def get_persistent_object_ids(jsonned_info):
	my_batch = jsonned_info[0]
	jsonned_batch = [json.loads(frame_ex_json) for frame_ex_json in my_batch]
	objects = [screen_out_statics(frame_ex['observed_objects']) for frame_ex in jsonned_batch]
	positions = [dict((obj[1], obj[2])for obj in fr_objs) for fr_objs in objects]
	obj_ids = [set([obj[1] for obj in fr_objs]) for fr_objs in objects]
	#really bad arbitrary choice breaking
	persistent_objects = sorted(list(set.intersection(* obj_ids)))[:10]
	persistents_positions = [[coord for pers in persistent_objects for coord in fr_dict[pers]] for fr_dict in positions]
	return np.array(persistents_positions).astype('float32')

def worldinfo_json_dealie(info_tens):
	info_tens.set_shape((256,))
	info_tens = tf.reshape(info_tens, (256, 1))
	retval = tf.py_func(get_persistent_object_ids, [info_tens], tf.float32)
	retval.set_shape((256,30))
	return retval

def test_pyfunc():
	dp = d.TFRecordsParallelByFileProvider(source_paths,
                                           postprocess={'worldinfo' : [(worldinfo_json_dealie, (), {})]},
                                           n_threads=1,
                                           batch_size=256,
                                           shuffle=True)
	sess = tf.Session()
	ops = dp.init_ops()
	queue = b.get_queue(ops[0], queue_type='random')
	enqueue_ops = []
	for op in ops:
	    enqueue_ops.append(queue.enqueue_many(op))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
	tf.train.start_queue_runners(sess=sess)
	inputs = queue.dequeue_many(256)

	for i in range(2):
	    res = sess.run(inputs)
	    # assert res['images'].shape == (20, 32, 32, 3)
	    # assert_equal(res['ids'], res['ids1'])
	    # assert_allclose(res['images'].mean(1).mean(1).mean(1), res['means'], rtol=1e-05)
	    print(res['worldinfo'][0])


def test_new_tfrecord():
	dp = PositionPredictionData(DATA_PATH, 256, n_threads = 1, shuffle = True, num_timesteps = 6, 
		max_num_objects = 100, max_num_actions = 10, positions_only = False, output_num_objects = 20, output_num_actions = 5)
	sess = tf.Session()
	ops = dp.init_ops()
	queue = b.get_queue(ops[0], queue_type='random')
	enqueue_ops = []
	for op in ops:
	    enqueue_ops.append(queue.enqueue_many(op))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
	tf.train.start_queue_runners(sess=sess)
	inputs = queue.dequeue_many(256)

	for i in range(10):
	    res = sess.run(inputs)
	    # assert res['images'].shape == (20, 32, 32, 3)
	    # assert_equal(res['ids'], res['ids1'])
	    # assert_allclose(res['images'].mean(1).mean(1).mean(1), res['means'], rtol=1e-05)
	    print(res.keys())
	    # print(res['positions_parsed'].shape)
	    # print(res['corresponding_actions'].shape)
	    print(np.linalg.norm(res['positions_parsed']))
	    print(np.linalg.norm(res['corresponding_actions']))
	    # print(np.linalg.norm(res['positions_parsed'][:, -30:]))



if __name__ == '__main__':
	test_new_tfrecord()






