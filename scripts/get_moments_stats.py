import h5py
import numpy as np
import json
import os
import tensorflow as tf
from PIL import Image
import time
import sys
import cPickle
from tqdm import trange, tqdm
from multiprocessing import Pool

#SECOND_DATASET_LOCS = ['dataset0', 'dataset1', 'dataset2', 'dataset3']
dataset = sys.argv[1]
PREFIX = int(sys.argv[2])
KEEP_EXISTING_FILES = True
SECOND_DATASET_LOCS = [dataset]
SECOND_DATASET_LOCS = [os.path.join('/mnt/fs1/datasets/four_world_dataset/', loc + '.hdf5') for loc in SECOND_DATASET_LOCS]
NEW_TFRECORD_TRAIN_LOC = '/mnt/fs1/datasets/four_world_dataset/new_tfdata_newobj'
NEW_TFRECORD_VAL_LOC = '/mnt/fs1/datasets/four_world_dataset/new_tfvaldata_newobj'
ATTRIBUTE_NAMES = ['images', 'normals', 'objects', 'depths', 'vels', 'accs', 'jerks',
        'images2', 'normals2', 'objects2', 'depths2', 'vels2', 'accs2', 'jerks2',
        'actions', 'actions2', 'object_data', 'object_data2', 
        'agent_data', 'is_not_teleporting', 'is_not_dropping', 'is_acting', 
        'is_not_waiting', 'reference_ids', 'is_object_there', 'is_object_in_view', 'is_object_in_view2']
HEIGHT = 128
WIDTH = 170

NUM_OBJECTS_EXPLICIT = 2
datum_shapes = [(HEIGHT, WIDTH, 3)] * 14 + [(9,), (7,), (NUM_OBJECTS_EXPLICIT, 21), (NUM_OBJECTS_EXPLICIT, 21), (6,), (1,), (1,), (1,), (1,), (2,), (NUM_OBJECTS_EXPLICIT,), (NUM_OBJECTS_EXPLICIT,), (NUM_OBJECTS_EXPLICIT,)]
ATTRIBUTE_SHAPES = dict(x for x in zip(ATTRIBUTE_NAMES, datum_shapes))

my_files = [h5py.File(loc, 'r') for loc in SECOND_DATASET_LOCS]
BATCH_SIZE = 256
NUM_BATCHES = len(my_files[0]['actions']) / 256

OTHER_CAM_ROT = np.array([[1., 0., 0.], [0., np.cos(np.pi / 6.), np.sin(np.pi / 6.)], [0., - np.sin(np.pi / 6.), np.cos(np.pi / 6.)]])
OTHER_CAM_POS = np.array([0., .5, 0.])
MY_CM_CENTER = np.array([HEIGHT / 2., WIDTH / 2.])
MY_F = np.diag([-134., 136.])

#TODO:
global LAST_both_ids
LAST_both_ids = None

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

def get_transformation_params(world_info):
        return np.array([world_info['avatar_right'], world_info['avatar_up'], world_info['avatar_forward']]), np.array(world_info['avatar_position'])

def transform_to_local(position, rot_mat, origin_pos = np.zeros(3, dtype = np.float32)):
        position = np.array(position)
        if(len(position) == 0): 
            position = np.array([0,0,0])
        return np.dot(rot_mat, (position - origin_pos))

def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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
	if len(split_type) == 3:
                return ':'.join([split_type[i] for i in [0, 2, 1]])
        else:
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

def get_ids_to_include(observed_objects, obj_arrays, actions, subset_indicators):
        global LAST_both_ids
        action_ids = [[idx] for idx in get_acted_ids(actions, subset_indicators)]
        retval = []
        for (frame_act_ids, frame_observed_objects) in zip(action_ids, observed_objects):
                other_obj_ids = [i for i in frame_observed_objects if i not in frame_act_ids and i != -1 and frame_observed_objects[i][4] == False]
                if None not in frame_act_ids:
                    if len(other_obj_ids) < 1:
                        if LAST_both_ids is None:
                            both_ids = frame_act_ids + [-1]
                        else:
                            both_ids = LAST_both_ids
                    else:
                        both_ids = frame_act_ids + other_obj_ids
                else:
                    if len(other_obj_ids) == 1:
                        if LAST_both_ids is None:
                            both_ids = other_obj_ids + [-1]
                        else:
                            both_ids = LAST_both_ids
                    else:
                        both_ids = other_obj_ids
                assert len(both_ids) == 2, 'More than one object found: ' + str(both_ids)
                LAST_both_ids = both_ids
                retval.append(both_ids)
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
                        pos[0] = float(HEIGHT) / float(HEIGHT) * pos[0]
                        pos[1] = float(WIDTH) / float(WIDTH) * pos[1]
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

def safe_dict_append(my_dict, my_key, my_val):
        if my_key in my_dict:
                my_dict[my_key].append(my_val)
        else:
                my_dict[my_key] = [my_val]

def get_reference_ids((file_num, bn)):
        if PREFIX is not None:
            file_num = PREFIX
        return [np.array([file_num, bn * BATCH_SIZE + i]).astype(np.int32) for i in range(BATCH_SIZE)]

def get_batch_data((file_num, bn), with_non_object_images = True):
        f = my_files[file_num]
        start = time.time()
        objects = f['objects1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        objects2 = f['objects2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        #if with_non_object_images:
                #images = f['images1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #normals = f['normals1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #images2 = f['images2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #normals2 = f['normals2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #depths = f['depths1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #depths2 = f['depths2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #vels = f['velocities1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #vels2 = f['velocities2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #accs = f['accelerations1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #accs2 = f['accelerations2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #jerks = f['jerks1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #jerks2 = f['jerks2'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        actions_raw = f['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        actions_raw = [json.loads(act) for act in actions_raw]
        indicators = get_subset_indicators(actions_raw)
        worldinfos = [json.loads(info) for info in f['worldinfo'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
        coordinate_transformations = [get_transformation_params(info) for info in worldinfos]
        actions, actions2 = get_actions(actions_raw, coordinate_transformations)
        object_data, is_object_there, is_object_in_view, object_data2, is_object_in_view2 = get_object_data(worldinfos, objects, objects2, actions_raw, indicators, coordinate_transformations)
        agent_data = get_agent_data(worldinfos)
        reference_ids = get_reference_ids((file_num, bn))
        to_ret = {'object_data' : object_data, 'object_data2' : object_data2}
        to_ret.update(indicators)
        to_ret = add_jerk(to_ret)
        return to_ret

def add_jerk(data):
    start = 0
    prev_pos = 0
    prev_vel = 0
    prev_acc = 0
    for o in ['object_data', 'object_data2']:
        data[o] = np.pad(data[o], [[0,0],[0,0],[0,9]], 'constant', 
                constant_values=np.array([[0,0],[0,0],[0,0]]))
        for t, no_teleport in enumerate(data['is_not_teleporting']):
            cur_pos = data[o][t,:,5:8]
            if t > start:
                cur_vel = cur_pos - prev_pos
            else:
                cur_vel = 0
            if t > start + 1:
                cur_acc = cur_vel - prev_vel
            else:
                cur_acc = 0
            if t > start + 2:
                jerk = cur_acc - prev_acc
            else:
                jerk = 0

            data[o][t,:,12:15] = cur_vel
            data[o][t,:,15:18] = cur_acc
            data[o][t,:,18:21] = jerk

            if not no_teleport[0]:
                start = t
                prev_pos = 0
                prev_vel = 0
                prev_acc = 0
            else:
                prev_pos = cur_pos
                prev_vel = cur_vel
                prev_acc = cur_acc
    return data

def remove_frames(data):
    to_remove = []
    for im, (in_view, in_view2) in enumerate(zip(data['is_object_in_view'], data['is_object_in_view2'])):
        # All objects have to be in view
        if np.sum(in_view) != len(in_view): # or np.sum(in_view2) != len(in_view2):
            to_remove.append(im)
    target_size = len(data['is_object_in_view']) - len(to_remove)
    for k in data.keys():
        if isinstance(data[k], list):
            data[k] = [d for i, d in enumerate(data[k]) if i not in to_remove]
        elif isinstance(data[k], np.ndarray):
            data[k] = np.delete(data[k], to_remove, axis=0)
        else:
            raise TypeError('Unknown type ' + str(type(data[k])))
    for k in data.keys():
        assert len(data[k]) == target_size, k + ' does not match target_size=' + str(target_size)
    return data

def write_stuff(batch_data, writers):
        start = time.time()
        batch_size = len(batch_data[batch_data.keys()[0]])
        for k, writer in writers.iteritems():
                #print(time.time() - start)
                #print('writing ' + k)
                for i in range(batch_size):
                        datum = tf.train.Example(features = tf.train.Features(feature = {k : _bytes_feature(batch_data[k][i].tostring())}))
                        writer.write(datum.SerializeToString())

def write_in_thread((file_num, batches, write_path, prefix)):
    if prefix is None:
        prefix = file_num
    # Open writers 
    output_file = os.path.join(write_path, 'stats', str(prefix) + ':' + str(batches[0]) + ':' + str(batches[-1]) + '.pkl')

    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
            os.makedirs(directory)

    write_dict = {'object_data': [], 'object_data2': [], 'is_not_teleporting': []}
    for i, batch in enumerate(batches):
        batch_data_dict = get_batch_data((file_num, batch), with_non_object_images = True)
        for k in batch_data_dict:
            if k in ['object_data', 'object_data2', 'is_not_teleporting']:
                write_dict[k].append(batch_data_dict[k])

        print(str(float(i) / len(batches) * 100.0) + ' %')
    for k in write_dict:
        write_dict[k] = np.array(write_dict[k])
        shape = write_dict[k].shape
        write_dict[k] = np.reshape(write_dict[k], list([shape[0] * shape[1]]) + list(shape[2:]))

    f = open(output_file, 'w')
    cPickle.dump(write_dict, f)

    return 0

def do_write(all_images = True):
	my_rng = np.random.RandomState(seed = 0)
        file_count = 0
	
	# create batch tasks
        write_tasks = []
        num_batches = 0
        for file_num in range(len(my_files)):
            write_task = []
            for batch_num in range(0, NUM_BATCHES, NUM_BATCHES):
                if my_rng.rand() > 0.1:
                    write_path = NEW_TFRECORD_TRAIN_LOC
                else:
                    write_path = NEW_TFRECORD_VAL_LOC
                write_task.append((file_num, range(batch_num, batch_num + NUM_BATCHES), write_path, PREFIX))
                num_batches += 1
            write_tasks.append(write_task)

        rs = []
        pools = []
        completed  = 0
        for write_task in write_tasks:
            for wt in write_task:
                write_in_thread(wt)
                completed += 1 
                print(str(float(completed) / num_batches * 100.0) + ' %')
        print('DONE')
        #for _ in tqdm(p.imap(write_in_thread, write_tasks), desc='batch', total=len(write_tasks)):
        #    pass

if __name__ == '__main__':
        do_write()
        for f in my_files:
                f.close()
