import h5py
import numpy as np
import json
import os
import tensorflow as tf
from PIL import Image
import time
import sys
import cPickle

DATASET_LOC = '/mnt/fs0/datasets/two_world_dataset/new_dataset.hdf5'
SECOND_DATASET_LOCS = ['dataset1', 'dataset2', 'dataset3', 'dataset5', 'dataset7']
SECOND_DATASET_LOCS = [os.path.join('/mnt/fs0/datasets/two_world_dataset/hdf5s', loc + '.hdf5') for loc in SECOND_DATASET_LOCS]
NEW_TFRECORD_TRAIN_LOC = '/mnt/fs0/datasets/two_world_dataset/new_tfdata_newobj'
NEW_TFRECORD_VAL_LOC = '/mnt/fs0/datasets/two_world_dataset/new_tfvaldata_newobj'
ATTRIBUTE_NAMES = ['images', 'normals', 'objects', 'images2', 'normals2', 'objects2', 'actions', 'actions2', 'object_data', 'object_data2', 'agent_data', 'is_not_teleporting', 'is_not_dropping', 'is_acting', 'is_not_waiting', 'reference_ids', 'is_object_there', 'is_object_in_view', 'is_object_in_view2']
WRITING_NOW = ['objects', 'objects2', 'actions', 'actions2', 'object_data', 'object_data2', 'agent_data', 'is_not_teleporting', 'is_not_dropping', 'is_acting', 'is_not_waiting', 'reference_ids', 'is_object_there', 'is_object_in_view', 'is_object_in_view2']
NEW_HEIGHT = 160
NEW_WIDTH = 375
# NEW_HEIGHT = 256
# NEW_WIDTH = 600
OLD_HEIGHT = 256
OLD_WIDTH = 600

for k in ATTRIBUTE_NAMES:
	if k not in WRITING_NOW:
		print('not writing ' + k)

JOBS_DIVISION = '/mnt/fs0/datasets/two_world_dataset/job_division.p'
NUM_OBJECTS_EXPLICIT = 11
datum_shapes = [(NEW_HEIGHT, NEW_WIDTH, 3)] * 6 + [(9,), (7,), (NUM_OBJECTS_EXPLICIT, 12), (NUM_OBJECTS_EXPLICIT, 12), (6,), (1,), (1,), (1,), (1,), (2,), (NUM_OBJECTS_EXPLICIT,), (NUM_OBJECTS_EXPLICIT,), (NUM_OBJECTS_EXPLICIT,)]
ATTRIBUTE_SHAPES = dict(x for x in zip(ATTRIBUTE_NAMES, datum_shapes))

my_files = [h5py.File(DATASET_LOC, 'r')] + [h5py.File(loc, 'r') for loc in SECOND_DATASET_LOCS]
BATCH_SIZE = 256
#big_buckets = ['ROLLY_THROW_OBJ:THROW_AT_OBJECT:0', 'ROLLY_WALL_THROW:THROW_BEHIND:0', 'OBJ_THROW_OBJ:THROW_AT_OBJECT:0', 'WALL_THROW:THROW_BEHIND:0']
#PREFIX_DIVISIONS = ['ONE_OBJ:', 'TABLE:', 'TABLE_CONTROLLED:', 'ROLLY_THROW_OBJ', 'ROLLY_WALL_THROW', 'OBJ_THROW_OBJ', 'WALL_THROW:', 'OBJ_ON_OBJ:', 'ONE_ROLLY:', 'ROLLY_ON_TABLE', 'ROLLY_ON_TABLE_CONTROLLED', 'ROLLY_ON_OBJ:', 'OBJ_ON_ROLLY:', 'ROLLY_ON_ROLLY:', ]
#pfx_fns = [fn + '.p' for fn in PREFIX_DIVISIONS]
#TYPES_DIVIDER_LOC = '/media/data3/new_dataset/first_sets.p'

all_those_last_names = set()
name_change_dict = {'LIFT' : 'FAST_LIFT', 'PUSH_ROT_NOSHAKE' : 'PUSH_ROT', 'PUSH_NOSHAKE' : 'LONG_PUSH', 'PUSH_DOWN' : 'DOWN_PUSH', 'FAST_LIFT_PUSH_ROT_NOSHAKE' : 'FAST_LIFT_PUSH_ROT', 'FAST_LIFT_NOSHAKE' : 'FAST_LIFT', 'PUSH_DOWN_NOSHAKE' : 'DOWN_PUSH'}


OTHER_CAM_ROT = np.array([[1., 0., 0.], [0., np.cos(np.pi / 6.), np.sin(np.pi / 6.)], [0., - np.sin(np.pi / 6.), np.cos(np.pi / 6.)]])
OTHER_CAM_POS = np.array([0., .5, 0.])
IM_HEIGHT = 160.
IM_WIDTH = 375.
MY_CM_CENTER = np.array([IM_HEIGHT / 2., IM_WIDTH / 2.])
MY_F = np.diag([-134., 136.])

def pos_to_screen_pos(pos, f, dist, cm_center):
	small_pos_array = np.array([pos[1], pos[0]])
	return np.dot(f, small_pos_array) / (pos[2] + dist) + cm_center

def std_pos_to_screen_pos(pos):
	return pos_to_screen_pos(pos, MY_F, 0.001, MY_CM_CENTER)


def rot_to_quaternion(rot_mat):
	tr = np.trace(rot_mat)
	if tr > 0:
		S = np.sqrt(tr + 1.) * 2.
		qw = .25 * S
		qx = (rot_mat[2,1] - rot_mat[1,2]) / S
		qy = (rot_mat[0,2] - rot_mat[2,0]) / S
		qz = (rot_mat[1,0] - rot_mat[0,1]) / S
	elif rot_mat[0,0] > rot_mat[1,1] and rot_mat[0,0] > rot_mat[2,2]:
		S = np.sqrt(1. + rot_mat[0,0] - rot_mat[1,1] - rot_mat[2,2]) * 2.
		qw = (rot_mat[2,1] - rot_mat[1,2]) / S
		qx = .25 * S
		qy = (rot_mat[0,1] + rot_mat[1,0]) / S
		qz = (rot_mat[0,2] + rot_mat[2,0]) / S
	elif rot_mat[1,1] > rot_mat[2,2]:
		S = np.sqrt(1. + rot_mat[1,1] - rot_mat[0,0] - rot_mat[2,2]) * 2.
		qw = (rot_mat[0,2] - rot_mat[2,0]) / S
		qx = (rot_mat[0,1] + rot_mat[1,0]) / S
		qy = .25 * S
		qz = (rot_mat[1,2] + rot_mat[2,1]) / S
	else:
		S = np.sqrt(1. + rot_mat[2,2] - rot_mat[0,0] - rot_mat[1,1]) * 2.
		qw = (rot_mat[1,0] - rot_mat[0,1]) / S
		qx = (rot_mat[0,2] + rot_mat[2,0]) / S
		qy = (rot_mat[1,2] + rot_mat[2,1]) / S
		qz = .25 * S
	return np.array([qw, qx, qy, qz])

OTHER_CAM_QUAT = rot_to_quaternion(OTHER_CAM_ROT)

def quat_mult(q1, q2):
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
	z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
	return w, x, y, z

def check_num_valid(my_hdf5):
	bn = 0
	valid = my_hdf5['valid']
	N = int(valid.shape[0] / BATCH_SIZE)
	for bn in range(0, N):
		if not valid[bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE].all():
			return bn
	return N


def get_transformation_params(world_info):
	return np.array([world_info['avatar_right'], world_info['avatar_up'], world_info['avatar_forward']]), np.array(world_info['avatar_position'])

def transform_to_local(position, rot_mat, origin_pos = np.zeros(3, dtype = np.float32)):
	position = np.array(position)
	return np.dot(rot_mat, (position - origin_pos))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def resize_images(images, use_nearest = False):
	if use_nearest:
		return [np.array(Image.fromarray(image).resize((NEW_WIDTH, NEW_HEIGHT), Image.NEAREST))
			for image in images]
	return [np.array(Image.fromarray(image).resize((NEW_WIDTH, NEW_HEIGHT), Image.ANTIALIAS))
		for image in images]

def remind_me_structure():
	print json.loads(f['worldinfo'][0]).keys()
	actions = [json.loads(act) for act in f['actions'][0:256]]
	for act in actions:
		print act

def print_agent_deets(bn):
	worldinfos = [json.loads(info) for info in f['worldinfo'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
	for info in worldinfos:
		print 'new frame'
		print info['avatar_position']
		print info['avatar_rotation']
		print info['avatar_right']
		print info['avatar_up']
		print info['avatar_forward']	

def get_action_descriptors(start_bn, end_bn):
	descriptors = set()
	for bn in range(start_bn, end_bn):
		actions = [json.loads(act) for act in f['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
		for act in actions:
			descriptors.add(act['action_type'])
	return descriptors


def process_action_type(actions):
	action_types = set([str(act['action_type']) for act in actions])
	to_pop = []
	for act_type in action_types:
		if 'WAIT' in act_type:
			to_pop.append(act_type)
		elif 'DROPPING' in act_type:
			to_pop.append(act_type)
		elif 'TELE' in act_type:
			to_pop.append(act_type)
	for act_type in to_pop:
		action_types.remove(act_type)
	assert len(action_types) == 1, len(action_types)
	my_action_type = action_types.pop()
	split_type = my_action_type.split(':')
	assert len(split_type) == 3 or len(split_type) == 4, len(split_type)
	split_type[-1] = name_change_dict.get(split_type[-1], split_type[-1])
	if len(split_type) == 3:
		all_those_last_names.add(split_type[2])
		return ':'.join([split_type[i] for i in [0, 2, 1]])
	else:
		all_those_last_names.add(split_type[3])
		return ':'.join([split_type[i] for i in [0, 2, 3, 1]])
	

def make_id_dict(observed_objects):
	return dict((int(o[1]), o) for o in observed_objects if not o[4])

def get_acted_ids(actions, subset_indicators):
	teleport_times = [idx for (idx, indicator) in enumerate(subset_indicators['is_not_teleporting']) if indicator[0] == 0]
	action_times = [idx for (idx, indicator) in enumerate(subset_indicators['is_acting']) if indicator[0] == 1]
	first_next_action_times = []
	objects_acted_on_after_teleport = []
	for t_0 in teleport_times:
		obj_acted_on = None
		for t in range(t_0, BATCH_SIZE):
			if t in action_times:
				first_next_action_times.append(t)
				obj_acted_on = int(actions[t]['actions'][0]['id'])
				break
		objects_acted_on_after_teleport.append(obj_acted_on)
	acted_on = []
	current_obj = None
	next_tele_time = teleport_times.pop(0) 
	for t in range(BATCH_SIZE):
		if t >= next_tele_time:
			current_obj = objects_acted_on_after_teleport.pop(0)
			if teleport_times:
				next_tele_time = teleport_times.pop(0)
			else:
				next_tele_time = BATCH_SIZE
		acted_on.append(current_obj)
	return acted_on

def get_centers_of_mass(obj_array, ids):
	oarray1 = 256**2 * obj_array[:, :, 0] + 256 * obj_array[:, :, 1] + obj_array[:, :, 2]
	cms = {}
	for idx in ids:
		if idx is None:
			continue
		xs, ys = (oarray1 == idx).nonzero()
		if len(xs) == 0:
			cms[idx] = None
		else:
			cms[idx] = np.round(np.array(zip(xs, ys)).mean(0))
	return cms

def get_most_pix_objects(frame_obj_array, frame_observed_objects, max_num_obj = 10):
	obj_array = frame_obj_array
	oarray1 = 256**2 * obj_array[:, :, 0] + 256 * obj_array[:, :, 1] + obj_array[:, :, 2]
	observed_objects = np.unique(oarray1)
	valid_o = [(idx, (oarray1 == idx).sum()) for idx in observed_objects if idx in frame_observed_objects and not frame_observed_objects[idx][4]]
	valid_o = sorted(valid_o, key = lambda (idx, amt) : amt)
	to_ret = []
	for _ in range(max_num_obj):
		if valid_o:
			to_ret.append(valid_o.pop(-1)[0])
		else:
			to_ret.append(None)
	return to_ret

def get_ids_to_include(observed_objects, obj_arrays, actions, subset_indicators):
	action_ids = [[idx] for idx in get_acted_ids(actions, subset_indicators)]
	retval = []
	for (obj_array, frame_observed_objects, frame_act_ids) in zip(obj_arrays, observed_objects, action_ids):
		retval.append(frame_act_ids + get_most_pix_objects(obj_array, frame_observed_objects, max_num_obj = NUM_OBJECTS_EXPLICIT - 1))
	return retval

def get_object_data(worldinfos, obj_arrays, obj_arrays2, actions, subset_indicators, coordinate_transformations):
	'''returns num_frames x num_objects x dim_data
	Object order: object acted on, second most important object, other objects in view ordered by distance, up to 10 objects.
	Data: id, pose, position, center of mass in image frame
	'''
	#TODO: rotate to agent frame
	observed_objects = [make_id_dict(info['observed_objects']) for info in worldinfos]
	ids_to_include = get_ids_to_include(observed_objects, obj_arrays, actions, subset_indicators)
	ret_list = []
	is_object_there = []
	is_object_in_view = []
	ret_list2 = []
	is_object_in_view2 = []
	for (frame_obs_objects, obj_array, obj_array2, frame_ids_to_include, (rot_mat, agent_pos)) in zip(observed_objects, obj_arrays, obj_arrays2, ids_to_include, coordinate_transformations):
		q_rot = rot_to_quaternion(rot_mat)
		centers_of_mass = get_centers_of_mass(obj_array, frame_ids_to_include)
		centers_of_mass2 = get_centers_of_mass(obj_array2, frame_ids_to_include)
		frame_data = []
		frame_data2 = []
		frame_obj_there_data = []
		frame_obj_in_view_data = []
		frame_obj_in_view_data2 = []
		for idx in frame_ids_to_include:
			if idx is None:
				obj_data = [np.array([-1.]).astype(np.float32)]
				obj_data2 = [np.array([-1.]).astype(np.float32)]
			else:
				obj_data = [np.array([idx])]
				obj_data2 = [np.array([idx])]
			if idx is None or idx not in frame_obs_objects:
				obj_data.append(np.zeros(11))
				obj_data2.append(np.zeros(11))
				frame_obj_there_data.append(0)
				frame_obj_in_view_data.append(0)
				frame_obj_in_view_data2.append(0)
			else:
				o = frame_obs_objects[idx]
				pose = quat_mult(q_rot, np.array(o[3]))
				obj_data.append(pose) #pose
				pose2 = quat_mult(OTHER_CAM_QUAT, pose)
				obj_data2.append(pose2)
				position = transform_to_local(o[2], rot_mat, agent_pos)
				position2 = transform_to_local(position, OTHER_CAM_ROT, OTHER_CAM_POS)
				obj_data.append(position) #3d position
				obj_data2.append(position2)
				screen_pos = std_pos_to_screen_pos(position) #screen position
				screen_pos2 = std_pos_to_screen_pos(position2)
				obj_data.append(screen_pos)
				obj_data2.append(screen_pos2)
				if centers_of_mass[idx] is None:
					frame_obj_in_view_data.append(0)
					obj_data.append(np.array([-100, -100.]).astype(np.float32))
				else:
					frame_obj_in_view_data.append(1)
					obj_data.append(np.array(centers_of_mass[idx]))
				if centers_of_mass2[idx] is None:
					frame_obj_in_view_data2.append(0)
					obj_data2.append(np.array([-100, -100.]).astype(np.float32))
				else:
					frame_obj_in_view_data2.append(1)
					obj_data2.append(np.array(centers_of_mass2[idx]))
				frame_obj_there_data.append(1)
			obj_data = np.concatenate(obj_data)
			obj_data2 = np.concatenate(obj_data2)
			frame_data.append(obj_data)
			frame_data2.append(obj_data2)
		is_object_there.append(np.array(frame_obj_there_data).astype(np.int32))
		is_object_in_view.append(np.array(frame_obj_in_view_data).astype(np.int32))
		is_object_in_view2.append(np.array(frame_obj_in_view_data2).astype(np.int32))
		frame_data = np.array(frame_data).astype(np.float32)
		frame_data2 = np.array(frame_data2).astype(np.float32)
		ret_list.append(frame_data)
		ret_list2.append(frame_data2)
	return ret_list, is_object_there, is_object_in_view, ret_list2, is_object_in_view2
				
	

def get_agent_data(worldinfos):
	'''returns num_frames x dim_data
	agent position, rotation...what are these other things?
	'''
	return [np.concatenate([np.array(info['avatar_position']), np.array(info['avatar_rotation'])]).astype(np.float32) 
for info in worldinfos]

def get_actions(actions, coordinate_transformations):
	'''returns num_frames x dim_data
	force, torque, position, id_acted_on (3d position?)
	'''
	ret_list = []
	ret_list2 = []
	for (act, (rot_mat, agent_pos)) in zip(actions, coordinate_transformations):
		if 'actions' in act and len(act['actions']) and 'teleport_to' not in act['actions'][0]:
			act_data = act['actions'][0]
			force = transform_to_local(act_data['force'], rot_mat)
			torque = transform_to_local(act_data['torque'], rot_mat)
			force2 = transform_to_local(force, OTHER_CAM_ROT)
			torque2 = transform_to_local(torque, OTHER_CAM_ROT)
			pos = np.array(act_data['action_pos'])
			if len(pos) != 2:
				pos = np.array([-100., -100.])
			pos[0] = float(NEW_HEIGHT) / float(OLD_HEIGHT) * pos[0]
			pos[1] = float(NEW_WIDTH) / float(OLD_WIDTH) * pos[1]
			idx = np.array([float(act_data['id'])])
			assert len(force) == 3 and len(torque) == 3 and len(pos) == 2 and len(idx) == 1, (len(force), len(torque), len(pos), len(idx))
			ret_list.append(np.concatenate([force, torque, pos, idx]).astype(np.float32))
			ret_list2.append(np.concatenate([force2, torque2, idx]).astype(np.float32))
		else:
			ret_list.append(np.zeros(9, dtype = np.float32)) 
			ret_list2.append(np.zeros(7, dtype = np.float32))
	return ret_list, ret_list2

def get_subset_indicators(actions):
	'''returns num_frames x 4 np array
	binary indicators 
	'''
	action_types = [act['action_type'] for act in actions]
	is_not_waiting = []
	is_not_dropping = []
	is_acting = []
	is_not_teleporting = []
	for act_type in action_types:
		not_waiting_now = int('WAIT' not in act_type)
		not_dropping_now = int('DROPPING' not in act_type)
		not_tele_now = int('TELE' not in act_type)
		acting_now = int(not_waiting_now + not_dropping_now + not_tele_now == 3)
		is_not_waiting.append(np.array([not_waiting_now]).astype(np.float32))
		is_not_dropping.append(np.array([not_dropping_now]).astype(np.float32))
		is_not_teleporting.append(np.array([not_tele_now]).astype(np.float32))
		is_acting.append(np.array([acting_now]).astype(np.float32))
	return {'is_not_waiting' : is_not_waiting, 'is_not_dropping' : is_not_dropping, 'is_acting' : is_acting, 'is_not_teleporting' : is_not_teleporting}

def figure_out_which_batches_go_where(start_bn, end_bn):
	'''Ahead-of-time processing that tells us which batches are going to be written in which file descriptor types.
	returns dict: processed action type -> batch numbers
	'''
	return_dict = {}
	for bn in range(start_bn, end_bn):
		actions = [json.loads(act) for act in f['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
		action_type = process_action_type(actions)
		if action_type in return_dict:
			return_dict[action_type].append(bn)
		else:
			return_dict[action_type] = [bn]
	return return_dict


def safe_dict_append(my_dict, my_key, my_val):
        if my_key in my_dict:
                my_dict[my_key].append(my_val)
        else:
                my_dict[my_key] = [my_val]

def index_types():
	valid_num_1 = 5810
	valid_num_rest = [int(check_num_valid(my_files[i]) / 70) * 70 for i in range(1, len(my_files))]
	valid_nums = [valid_num_1] + valid_num_rest
	print('total number of batches ' + str(sum(valid_nums)))
	print('per file: ')
	print(valid_nums)
	return_dict = {}
	for i in range(len(my_files)):
		for bn in range(valid_nums[i]):
			actions = [json.loads(act) for act in my_files[i]['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
			action_type = process_action_type(actions)
			safe_dict_append(return_dict, action_type, (i, bn))
	return return_dict

def get_job_buckets(num_buckets):
	print('indexing types')
	types_dict = index_types()
	print('placing types')
	type_and_num = [(k, len(v)) for (k, v) in types_dict.iteritems()]
	type_and_num = sorted(type_and_num, key = lambda (arg_1, arg_2) : - arg_2)
	buckets = [[] for _ in range(num_buckets)]
	bucket_weights = [0 for _ in range(num_buckets)]
	for (my_type, weight) in type_and_num:
		i = bucket_weights.index(min(bucket_weights))
		buckets[i].append(my_type)
		bucket_weights[i] += weight
	type_and_num_dict = dict(x for x in type_and_num)
	for i in range(num_buckets):
		should_be_total_weight = sum([type_and_num_dict[k] for k in buckets[i]])
		assert should_be_total_weight == bucket_weights[i]
		print('Bucket ' + str(i))
		print(bucket_weights[i])
	with open(JOBS_DIVISION, 'w') as stream:
		cPickle.dump((buckets, types_dict), stream)
	return types_dict, buckets, type_and_num, bucket_weights

def make_train_val_splits():
	with open(JOBS_DIVISION) as stream:
		buckets, types_dict = cPickle.load(stream)
	keys_shortened = list(set([':'.join(k.split(':')[:-1]) for k in types_dict]))
	shortened_summary = dict((k_start, sum([len(v) for k, v in types_dict.iteritems() if k.startswith(k_start)])) for k_start in keys_shortened)
	return shortened_summary


def get_reference_ids((file_num, bn)):
	return [np.array([file_num, bn * BATCH_SIZE + i]).astype(np.int32) for i in range(BATCH_SIZE)]

def get_batch_data((file_num, bn), with_non_object_images = True):
	f = my_files[file_num]
	start = time.time()
	print('reading objects')
	objects = f['objects'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	print(time.time() - start)
	print('resizing objects')
	objects = resize_images(objects, use_nearest = True)
	print(time.time() - start)
	print('reading objects2')
	objects2 = f['objects2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	print(time.time() - start)
	print('resizing objects2')
	objects2 = resize_images(objects2, use_nearest = True)
	print(time.time() - start)
	if with_non_object_images:
		print('reading images')
		images = f['images'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
		print(time.time() - start)
		print('resizing images')
		images = resize_images(images)
		print(time.time() - start)
		print('reading normals')
		normals = f['normals'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
		print(time.time() - start)
		print('resizing normals')
		normals = resize_images(normals)
		print(time.time() - start)
		print('reading images2')
		images2 = f['images2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
		print(time.time() - start)
		print('resizing images2')
		images2 = resize_images(images2)
		print(time.time() - start)	
		print('reading normals2')
		normals2 = f['normals2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
		print(time.time() - start)
		print('resizing normals2')
		normals2 = resize_images(normals2)
		print(time.time() - start)
	print('little processing')
	actions_raw = f['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	actions_raw = [json.loads(act) for act in actions_raw]
	indicators = get_subset_indicators(actions_raw)
	worldinfos = [json.loads(info) for info in f['worldinfo'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
	coordinate_transformations = [get_transformation_params(info) for info in worldinfos]
	actions, actions2 = get_actions(actions_raw, coordinate_transformations)
	object_data, is_object_there, is_object_in_view, object_data2, is_object_in_view2 = get_object_data(worldinfos, objects, objects2, actions_raw, indicators, coordinate_transformations)
	agent_data = get_agent_data(worldinfos)
	reference_ids = get_reference_ids((file_num, bn))
	to_ret = {'objects' : objects,
		'objects2': objects2,
		'actions' : actions, 'actions2' : actions2, 'object_data' : object_data, 'object_data2' : object_data2, 'agent_data' : agent_data, 'reference_ids' : reference_ids,
		'is_object_there' : is_object_there, 'is_object_in_view' : is_object_in_view, 'is_object_in_view2' : is_object_in_view2}
	if with_non_object_images:
		to_ret.update({'images' : images, 'normals' : normals, 'images2' : images2, 'normals2' : normals2})
	to_ret.update(indicators)
	for i in range(BATCH_SIZE):
		for k in to_ret:
			assert to_ret[k][i].shape == ATTRIBUTE_SHAPES[k], (k, to_ret[k][i].shape, ATTRIBUTE_SHAPES[k])
	print(time.time() - start)
	return to_ret

def write_stuff(batch_data, writers):
	start = time.time()
	for k, writer in writers.iteritems():
		print(time.time() - start)
		print('writing ' + k)
		for i in range(BATCH_SIZE):
			datum = tf.train.Example(features = tf.train.Features(feature = {k : _bytes_feature(batch_data[k][i].tostring())}))
			writer.write(datum.SerializeToString())

def do_write(job_bucket_num, done_fn, all_images = True):
	if not os.path.exists(NEW_TFRECORD_TRAIN_LOC):
		os.mkdir(NEW_TFRECORD_TRAIN_LOC)
	if not os.path.exists(NEW_TFRECORD_VAL_LOC):
		os.mkdir(NEW_TFRECORD_VAL_LOC)

	done_fn = os.path.join(NEW_TFRECORD_TRAIN_LOC, done_fn)

	my_rng = np.random.RandomState(seed = job_bucket_num)

	if os.path.exists(done_fn):
		with open(done_fn) as stream:
			done_dict = cPickle.load(stream)
	else:
		done_dict = {}
	
	for nm in ATTRIBUTE_NAMES:
		write_dir_train = os.path.join(NEW_TFRECORD_TRAIN_LOC, nm)
		write_dir_val = os.path.join(NEW_TFRECORD_VAL_LOC, nm)
		if not os.path.exists(write_dir_train):
			os.mkdir(write_dir_train)
		if not os.path.exists(write_dir_val):
			os.mkdir(write_dir_val)
		
	with open(JOBS_DIVISION) as stream:
		buckets, types_dict = cPickle.load(stream)
	my_bucket = buckets[job_bucket_num]
	file_count = 0
	for batch_type in my_bucket:
		print 'Writing type ' + batch_type
		writers = None
		for (num_written, bn) in enumerate(types_dict[batch_type]):
			print 'writing bn ' + str(bn)
			if num_written % 4 == 0:
				print('getting writers')
				if writers is not None:
					for writer in writers.values():
						writer.close()
				for_training = my_rng.rand()
				if for_training > .1:
					write_path = NEW_TFRECORD_TRAIN_LOC
				else:
					write_path = NEW_TFRECORD_VAL_LOC
				output_files = [os.path.join(write_path, attr_name, batch_type + ':' + str(file_count) + '.tfrecords') for attr_name in WRITING_NOW]
				writers = dict((attr_name, tf.python_io.TFRecordWriter(file_name)) for (attr_name, file_name) in zip(WRITING_NOW, output_files))
				file_count += 1
			batch_data_dict = get_batch_data(bn, with_non_object_images = all_images)
			write_stuff(batch_data_dict, writers)
			safe_dict_append(done_dict, batch_type, bn)
			with open(done_fn, 'w') as stream:
				cPickle.dump(done_dict, stream)
		if writers is not None:
			for writer in writers.values():
				writer.close()
			


#remind_me_deets()

#which_go_where =  figure_out_which_batches_go_where(0, 1000)

#print all_those_last_names

#print_agent_deets(1)

#my_dict = divide_by_types()
#arg_num = int(sys.argv[1])

#do_write(arg_num, 'done_bucket_' + str(arg_num) + '.p')

#types_dict, buckets, type_and_num, bucket_weights = get_job_buckets(8)

#summary_dict = make_train_val_splits()

if __name__ == '__main__':
	arg_num = int(sys.argv[1])
	do_write(arg_num, 'done_bucket_' + str(arg_num) + '.p', all_images = False)
	for f in my_files:
		f.close()
