import h5py
import numpy as np
import json
import os
import tensorflow as tf
from PIL import Image

DATASET_LOC = '/media/data3/new_dataset/new_dataset.hdf5'
NEW_TFRECORD_TRAIN_LOC = '/media/data3/new_dataset/tfdata'
ATTRIBUTE_NAMES = ['images', 'normals', 'objects', 'images2', 'normals2', 'objects2', 'actions', 'object_data', 'agent_data', 'is_not_teleporting', 'is_not_dropping', 'is_acting', 'is_not_waiting']
NEW_HEIGHT = 256
NEW_WIDTH = 256

f = h5py.File(DATASET_LOC, 'r')
BATCH_SIZE = 256


all_those_last_names = set()
name_change_dict = {'LIFT' : 'FAST_LIFT', 'PUSH_ROT_NOSHAKE' : 'PUSH_ROT', 'PUSH_NOSHAKE' : 'LONG_PUSH', 'PUSH_DOWN' : 'DOWN_PUSH', 'FAST_LIFT_PUSH_ROT_NOSHAKE' : 'FAST_LIFT_PUSH_ROT', 'FAST_LIFT_NOSHAKE' : 'FAST_LIFT', 'PUSH_DOWN_NOSHAKE' : 'DOWN_PUSH'}

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def resize_images(images):
	return [np.array(Image.fromarray(image).resize((NEW_HEIGHT, NEW_WIDTH), Image.BICUBIC))
		for image in images]

def remind_me_structure():
	print json.loads(f['worldinfo'][0]).keys()
	actions = [json.loads(act) for act in f['actions'][0:256]]
	for act in actions:
		print act


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
			cms[idx] = np.array([-100., -100.])
		else:
			cms[idx] = np.round(np.array(zip(xs, ys)).mean(0))
	return cms


def get_ids_to_include(observed_objects, obj_arrays, actions, subset_indicators):
	#TODO add other objects
	retval = [[idx] for idx in get_acted_ids(actions, subset_indicators)]
	return retval

def get_object_data(worldinfos, obj_arrays, actions, subset_indicators):
	'''returns num_frames x num_objects x dim_data
	Object order: object acted on, second most important object, other objects in view ordered by distance, up to 10 objects.
	Data: id, pose, position, center of mass in image frame
	'''
	#TODO: rotate to agent frame
	observed_objects = [make_id_dict(info['observed_objects']) for info in worldinfos]
	ids_to_include = get_ids_to_include(observed_objects, obj_arrays, actions, subset_indicators)
	ret_list = []
	for (frame_obs_objects, obj_array, frame_ids_to_include) in zip(observed_objects, obj_arrays, ids_to_include):
		centers_of_mass = get_centers_of_mass(obj_array, frame_ids_to_include)
		frame_data = []
		for idx in frame_ids_to_include:
			obj_data = [np.array([idx])]
			if idx is None:
				obj_data.append(np.zeros(9))
			else:
				o = frame_obs_objects[idx]
				obj_data.append(np.array(o[3])) #pose
				obj_data.append(np.array(o[2])) #3d position
				obj_data.append(np.array(centers_of_mass[idx]))
			obj_data = np.concatenate(obj_data)
			frame_data.append(obj_data)
		frame_data = np.array(frame_data).astype('float32')
		ret_list.append(frame_data)
	return ret_list
				
	

def get_agent_data(worldinfos):
	'''returns num_frames x dim_data
	agent position, rotation...what are these other things?
	'''
	return [np.concatenate([np.array(info['avatar_position']), np.array(info['avatar_rotation'])]).astype('float32') 
for info in worldinfos]

def get_actions(actions):
	'''returns num_frames x dim_data
	force, torque, position, id_acted_on (3d position?)
	'''
	ret_list = []
	for act in actions:
		if 'actions' in act and len(act['actions']) and 'teleport_to' not in act['actions'][0]:
			act_data = act['actions'][0]
			force = act_data['force']
			torque = act_data['torque']
			pos = act_data['action_pos']
			idx = [float(act_data['id'])]
			ret_list.append(np.concatenate([force, torque, pos, idx]).astype('float32'))
		else:
			ret_list.append(np.zeros(9, dtype = np.float32)) 
	return ret_list
			

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

def get_batch_data(bn):
	images = f['images'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	images = resize_images(images)
	objects = f['objects'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	objects = resize_images(objects)
	normals = f['normals'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	normals = resize_images(normals)
	images2 = f['images2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	images2 = resize_images(images2)
	objects2 = f['objects2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	objects2 = resize_images(objects2)
	normals2 = f['normals2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	normals2 = resize_images(normals2)
	actions_raw = f['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
	actions_raw = [json.loads(act) for act in actions_raw]
	actions = get_actions(actions_raw)
	indicators = get_subset_indicators(actions_raw)
	worldinfos = [json.loads(info) for info in f['worldinfo'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
	object_data = get_object_data(worldinfos, objects, actions_raw, indicators)
	agent_data = get_agent_data(worldinfos)
	to_ret = {'images' : images, 'objects' : objects, 'normals' : normals,
		'images2' : images2, 'objects2': objects2, 'normals2' : normals2,
		'actions' : actions, 'object_data' : object_data, 'agent_data' : agent_data}
	to_ret.update(indicators)
	return to_ret

def write_stuff(batch_data, writers):
	for k, writer in writers.iteritems():
		for i in range(BATCH_SIZE):
			datum = tf.train.Example(features = tf.train.Features(feature = {k : _bytes_feature(batch_data[k][i].tostring())}))
			writer.write(datum.SerializeToString())

def do_write(start_bn, end_bn):
	if not os.path.exists(NEW_TFRECORD_TRAIN_LOC):
		os.mkdir(NEW_TFRECORD_TRAIN_LOC)

	for nm in ATTRIBUTE_NAMES:
		write_dir = os.path.join(NEW_TFRECORD_TRAIN_LOC, nm)
		if not os.path.exists(write_dir):
			os.mkdir(write_dir)

	batch_type_dict = figure_out_which_batches_go_where(start_bn, end_bn) # might want to do this once ahead of time, if there are multiple runs planned
	file_count = 0
	for batch_type, bns in batch_type_dict.iteritems():
		print 'Writing type ' + batch_type
		writers = None
		for (num_written, bn) in enumerate(bns):
			print 'writing bn ' + str(bn)		
			if num_written % 4 == 0:
				if writers is not None:
					for writer in writers.values():
						writer.close()
				output_files = [os.path.join(NEW_TFRECORD_TRAIN_LOC, attr_name, batch_type + ':' + str(file_count) + '.tfrecords') 
					for attr_name in ATTRIBUTE_NAMES]
				writers = dict((attr_name, tf.python_io.TFRecordWriter(file_name)) for (attr_name, file_name) in zip(ATTRIBUTE_NAMES, output_files))
				file_count += 1
			batch_data_dict = get_batch_data(bn)
			write_stuff(batch_data_dict, writers)
		for writer in writers.values():
			writer.close()
			

#which_go_where =  figure_out_which_batches_go_where(0, 1000)

#print all_those_last_names


do_write(0, 200)
