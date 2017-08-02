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
GRID_DIM = 32
OUTPUT_GRID = True
assert GRID_DIM <= 256, 'extend data type from uint8 to uint16!'
KEEP_EXISTING_FILES = True
SECOND_DATASET_LOCS = [dataset]
SECOND_DATASET_LOCS = [os.path.join('/mnt/fs1/datasets/eight_world_dataset/', loc + '.hdf5') for loc in SECOND_DATASET_LOCS]
NEW_TFRECORD_TRAIN_LOC = '/mnt/fs1/datasets/eight_world_dataset/tfdata'
NEW_TFRECORD_VAL_LOC = '/mnt/fs1/datasets/eight_world_dataset/tfvaldata'
ATTRIBUTE_NAMES = ['images', 'objects', 'depths', 'vels', 'vels_curr', 
        'actions', 'object_data', 'agent_data', 'is_not_teleporting', 'is_not_dropping', 'is_acting', 'is_not_waiting', 'reference_ids', 'is_object_there', 'is_object_in_view', 'max_coordinates', 'min_coordinates', 'particles'] 
for k in ['sparse_coordinates', 'sparse_particles', 'sparse_length', 'sparse_shape']:
    ATTRIBUTE_NAMES.append(k + '_' + str(GRID_DIM)) 
HEIGHT = 128
WIDTH = 170

MAX_PARTICLES = 3456
NUM_OBJECTS_EXPLICIT = 2
datum_shapes = [(HEIGHT, WIDTH, 3)] * 5 + [(2, 9), (NUM_OBJECTS_EXPLICIT, 14), (6,), (1,), (1,), (1,), (1,), (2,), (NUM_OBJECTS_EXPLICIT,), (NUM_OBJECTS_EXPLICIT,), (3,), (3,), (MAX_PARTICLES * 7,), (MAX_PARTICLES, 3), (MAX_PARTICLES, 15), (1,), (4,)]

if OUTPUT_GRID:
    ATTRIBUTE_NAMES.append('grid_' + str(GRID_DIM))
    datum_shapes.append((GRID_DIM, GRID_DIM, GRID_DIM, 15))

ATTRIBUTE_SHAPES = dict(x for x in zip(ATTRIBUTE_NAMES, datum_shapes))

#TODO CREATE META.PKL OUT OF SHAPES!!!
my_files = [h5py.File(loc, 'r') for loc in SECOND_DATASET_LOCS]
BATCH_SIZE = 256
NUM_BATCHES = len(my_files[0]['actions']) / 256

OTHER_CAM_ROT = np.array([[1., 0., 0.], [0., np.cos(np.pi / 6.), np.sin(np.pi / 6.)], [0., - np.sin(np.pi / 6.), np.cos(np.pi / 6.)]])
OTHER_CAM_POS = np.array([0., 0.5, 0.]) #TODO Shouldn't this be 10?
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
        if len(teleport_times) == 0:
            raise ValueError('No teleport times')
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
                            both_ids = frame_act_ids + [None]
                        else:
                            #both_ids = LAST_both_ids
                            both_ids = frame_act_ids + [None]
                    else:
                        both_ids = frame_act_ids + other_obj_ids
                else:
                    if len(other_obj_ids) == 1:
                        if LAST_both_ids is None:
                            both_ids = other_obj_ids + [None]
                        else:
                            #both_ids = LAST_both_ids
                            both_ids = other_obj_ids + [None]
                    else:
                        both_ids = other_obj_ids
                if len(both_ids) != 2:
                    raise ValueError('Wrong action length')
                assert len(both_ids) == 2, 'More than one object found: ' + str(both_ids)
                LAST_both_ids = both_ids
                retval.append(both_ids)
        return retval

def get_object_data(worldinfos, obj_arrays, actions, subset_indicators, coordinate_transformations):
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
        for (frame_obs_objects, obj_array, frame_ids_to_include, (rot_mat, agent_pos)) in zip(observed_objects, obj_arrays, ids_to_include, coordinate_transformations):
                q_rot = rot_to_quaternion(rot_mat)
                centers_of_mass = get_centers_of_mass(obj_array, frame_ids_to_include)
                frame_data = []
                frame_obj_there_data = []
                frame_obj_in_view_data = []
                for idx in frame_ids_to_include:
                        if idx is None:
                                obj_data = [np.array([-1.]).astype(np.float32)]
                        else:
                                obj_data = [np.array([idx])]
                        if idx is None or idx not in frame_obs_objects:
                                obj_data.append(np.zeros(13))
                                frame_obj_there_data.append(0)
                                frame_obj_in_view_data.append(0)
                        else:
                                o = frame_obs_objects[idx]
                                pose = quat_mult(q_rot, np.array(o[3]))
                                pose2 = quat_mult(OTHER_CAM_QUAT, pose)
                                obj_data.append(pose2)
                                position = transform_to_local(o[2], rot_mat, agent_pos)
                                position2 = transform_to_local(position, OTHER_CAM_ROT, OTHER_CAM_POS)
                                obj_data.append(position2) #3d position
                                screen_pos2 = std_pos_to_screen_pos(position2) #screen position
                                obj_data.append(screen_pos2)
                                if centers_of_mass[idx] is None:
                                        frame_obj_in_view_data.append(0)
                                        obj_data.append(np.array([-100, -100.]).astype(np.float32))
                                else:
                                        frame_obj_in_view_data.append(1)
                                        obj_data.append(np.array(centers_of_mass[idx]))
                                frame_obj_there_data.append(1)
                                assert o[8] != -1, 'not a flex object! %d' % o[8]
                                assert o[9] > 0, 'number of particles <= 0: %d' % o[9]
                                obj_data.append([o[8]])
                                obj_data.append([o[9]])
                        obj_data = np.concatenate(obj_data)
                        frame_data.append(obj_data)
                is_object_there.append(np.array(frame_obj_there_data).astype(np.int32))
                is_object_in_view.append(np.array(frame_obj_in_view_data).astype(np.int32))
                frame_data = np.array(frame_data).astype(np.float32)
                ret_list.append(frame_data)
        return ret_list, is_object_there, is_object_in_view

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
        for (act, (rot_mat, agent_pos)) in zip(actions, coordinate_transformations):
                if 'actions' in act and len(act['actions']) and 'teleport_to' not in act['actions'][0]:
			fr_act_data = []
			for act_data in act['actions']:
                                if(len(act_data['force']) == 0):
                                    force2 = force = np.array([0,0,0])
                                else:
                                    force2 = force = act_data['force']
                                if(len(act_data['torque']) == 0):
                                    torque2 = torque = np.array([0,0,0])
                                else:
                                    torque2 = torque = act_data['torque']
                                # TODO: UNCOMMENT THIS IF RELATIVE TO AGENT
                                #force = transform_to_local(force, rot_mat)
                        	#torque = transform_to_local(torque, rot_mat)
                        	#force2 = transform_to_local(force, OTHER_CAM_ROT)
                        	#torque2 = transform_to_local(torque, OTHER_CAM_ROT)
                        	pos = np.array(act_data['action_pos'])
                        	if len(pos) != 2:
                                	pos = np.array([-100., -100.])
                        	pos[0] = float(HEIGHT) / float(HEIGHT) * pos[0]
                        	pos[1] = float(WIDTH) / float(WIDTH) * pos[1]
                        	idx = np.array([float(act_data['id'])])
                        	assert len(force) == 3 and len(torque) == 3 and len(pos) == 2 and len(idx) == 1, (len(force), len(torque), len(pos), len(idx))
                        	fr_act_data.append(np.concatenate([force2, torque2, pos, idx]).astype(np.float32))
			if len(act['actions']) == 1:
				fr_act_data.append(np.zeros(9, dtype = np.float32))
			fr_act_data = np.array(fr_act_data)
			ret_list.append(fr_act_data)
                else:
                        ret_list.append(np.zeros((2,9), dtype = np.float32))
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

def safe_dict_append(my_dict, my_key, my_val):
        if my_key in my_dict:
                my_dict[my_key].append(my_val)
        else:
                my_dict[my_key] = [my_val]

def get_reference_ids((file_num, bn)):
        if PREFIX is not None:
            file_num = PREFIX
        return [np.array([file_num, bn * BATCH_SIZE + i]).astype(np.int32) for i in range(BATCH_SIZE)]

def create_occupancy_grid(particles, object_data, actions, grid_dim):
    state_dim = 7 + 6 + 2 #pos, mass, vel, force, torque, 0/1 id, id
    if isinstance(grid_dim, list) or isinstance(grid_dim, np.ndarray):
        assert len(grid_dim) == 3, 'len(grid_dim) = %d' % len(grid_dim)
    else:
        grid_dim = [grid_dim] * 3
    grid_dim = np.array(grid_dim).astype(np.int32)
    grid = np.zeros([BATCH_SIZE] + list(grid_dim) + [state_dim])
    # create particle state vectors
    particles = np.reshape(particles, [BATCH_SIZE, particles.shape[1] / 7, 7])
    object_data = np.array(object_data) #id: 0, num_particles: 13
    ids = object_data[:, :, 0]
    n_particles = object_data[:, :, 13].astype(np.int32) / 7
    actions = np.array(actions)
    all_particles = np.sum(n_particles[0])
    for n in n_particles:
        assert np.sum(n) == all_particles
    #assert (np.sum(n_particles, axis=0) == n_particles[0] * BATCH_SIZE).all(), \
    #        str(np.sum(n_particles, axis=0)) + ' != ' + str(n_particles[0]*BATCH_SIZE)\
    #        + ' | ' + str(ids)
    particles = particles[:, :all_particles]
    # Assemble the state by adding forces, torques and ids
    # map ids to 0 = first object and 1 = second object such that 
    # taking the mean over the ids makes sense, i.e. that 
    # the mean equals to the part of the voxel that belongs to the second object
    particle_ids = np.zeros((particles.shape[0], particles.shape[1], 2))
    particle_actions = np.zeros((particles.shape[0], particles.shape[1], 6))
    for batch_index, batch_n_particles in enumerate(n_particles):
        offset = 0
        for n_index, n in enumerate(batch_n_particles):
            assert len(n.shape) == 0,  'len(n.shape) = %d, n = %d' % (len(n.shape), n)
            assert ids[batch_index, n_index] not in [-1, 0]
            assert ids[batch_index, n_index] in [23, 24]
            particle_ids[batch_index, offset:offset+n, 0] = \
                    0 if ids[batch_index, n_index] == 23 else 1
            particle_ids[batch_index, offset:offset+n, 1] = ids[batch_index, n_index]
            if actions[batch_index, n_index, 8] not in [-1, 0]:
                assert ids[batch_index, n_index] == actions[batch_index, n_index, 8]
                particle_actions[batch_index, offset:offset+n, :] = actions[batch_index, n_index, 0:6]
            offset += n
    states = np.concatenate([particles, particle_actions, particle_ids], axis = -1)

    # create indices for occupancy grid
    indices = particles[:,:,0:3]
    max_coordinates = np.amax(indices, axis=1)
    min_coordinates = np.amin(indices, axis=1)
    indices = (indices - min_coordinates[:,np.newaxis,:]) / (max_coordinates[:,np.newaxis,:] - min_coordinates[:,np.newaxis,:])
    indices = np.round(indices * (grid_dim - 1))
    batch_indices = np.reshape(np.tile(np.arange(BATCH_SIZE)[:,np.newaxis],
        [1,particles.shape[1]]), [BATCH_SIZE, particles.shape[1], 1])
    coordinates = np.concatenate([batch_indices, indices], axis=-1).astype(np.int32)

    # melt together points at the same index, if two different object are at the same position, reflect that in the relation grid, and set id to be belonging to both
    sparse_coordinates = []
    sparse_particles = []
    sparse_length = []
    sparse_shape = []
    for batch, (batch_coordinates, batch_states) in enumerate(zip(coordinates, states)):
        particle_coordinates = \
                [tuple(particle_coordinate) for particle_coordinate in batch_coordinates]
        sorted_particle_coordinates_indices = sorted(range(len(particle_coordinates)), \
                key=particle_coordinates.__getitem__)
        #sorted_particle_coordinates = sorted(particle_coordinates)
        sorted_particle_coordinates = [particle_coordinates[i] \
                for i in sorted_particle_coordinates_indices]
        unique_coordinates, idx_start, counts = np.unique(sorted_particle_coordinates, \
                return_index=True, return_counts=True, axis=0)
        identical_coordinates_sets = np.split(sorted_particle_coordinates_indices, \
                idx_start[1:])
        # mean position, sum mass, mean velocity, mean force, mean torque, mean ids
        particle_states = np.array([np.sum(batch_states[particle_coordinate], \
                axis=0) / counts[i] for i, particle_coordinate in enumerate(identical_coordinates_sets)])
        particle_states[:,3] *= counts # sum mass 
        particle_states[:,-1] *= counts # sum real ids, but not binary ones
        # store data for sparse tensor
        sparse_coordinates.append(unique_coordinates)
        sparse_particles.append(particle_states)
        sparse_length.append([len(sparse_coordinates)])
        sparse_shape.append(grid.shape[1:])

        # fill grid
        grid[unique_coordinates[:,0], \
             unique_coordinates[:,1], \
             unique_coordinates[:,2], \
             unique_coordinates[:,3]] = particle_states

    # pad with zeros
    tmp = np.zeros((BATCH_SIZE, MAX_PARTICLES, 3))
    for batch, sc in enumerate(sparse_coordinates):
        tmp[batch, 0:sc.shape[0]] = sc[:,0:3]
    sparse_coordinates = np.array(tmp).astype(np.uint8)
    tmp = np.zeros((BATCH_SIZE, MAX_PARTICLES, 15))
    for batch, sp in enumerate(sparse_particles):
        tmp[batch, 0:sp.shape[0]] = sp
    sparse_particles = np.array(tmp).astype(np.float32)
    # format
    grid = np.array(grid).astype(np.float16) #TODO USE HALF PRECISION AND NORMALIZE DATA!!!
    sparse_length = np.array(sparse_length).astype(np.int32)
    sparse_shape = np.array(sparse_shape).astype(np.int32)
    max_coordinates = np.array(max_coordinates).astype(np.float32)
    min_coordinates = np.array(min_coordinates).astype(np.float32)
    return grid, sparse_coordinates, sparse_particles, sparse_length, sparse_shape, max_coordinates, min_coordinates

def get_batch_data((file_num, bn), with_non_object_images = True):
        f = my_files[file_num]
        start = time.time()
        objects = f['objects1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        if with_non_object_images:
                images = f['images1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                depths = f['depths1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                vels = f['velocities1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                vels_curr = f['velocities_current1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                unpadded_particles = f['particles'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
                #pad particles with zeros to 10,000 * 7 dim
                particles = np.zeros((BATCH_SIZE, MAX_PARTICLES * 7))
                for p_index, p in enumerate(unpadded_particles):
                    particles[p_index,:p.shape[0]] = p

        actions_raw = f['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        actions_raw = [json.loads(act) for act in actions_raw]
        indicators = get_subset_indicators(actions_raw)
        worldinfos = [json.loads(info) for info in f['worldinfo'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
        coordinate_transformations = [get_transformation_params(info) for info in worldinfos]
        actions = get_actions(actions_raw, coordinate_transformations)
        object_data, is_object_there, is_object_in_view = get_object_data(worldinfos, objects, actions_raw, indicators, coordinate_transformations)
        grid, sparse_coordinates, sparse_particles, sparse_length, sparse_shape, max_coordinates, min_coordinates = create_occupancy_grid(particles, object_data, actions, GRID_DIM)
        agent_data = get_agent_data(worldinfos)
        reference_ids = get_reference_ids((file_num, bn))
        to_ret = {'objects' : objects, 'depths': depths, 'vels': vels, 'vels_curr': vels_curr, 'actions' : actions, 'object_data' : object_data, 'agent_data' : agent_data, 'reference_ids' : reference_ids, 'is_object_there' : is_object_there, 'is_object_in_view' : is_object_in_view, 'particles': particles, 'max_coordinates': max_coordinates, 'min_coordinates': min_coordinates}
        # add grid data
        if OUTPUT_GRID:
            to_ret['grid' + '_' + str(GRID_DIM)] = grid
        to_ret['sparse_coordinates' + '_' + str(GRID_DIM)] = sparse_coordinates
        to_ret['sparse_particles' + '_' + str(GRID_DIM)] = sparse_particles
        to_ret['sparse_length' + '_' + str(GRID_DIM)] = sparse_length
        to_ret['sparse_shape' + '_' + str(GRID_DIM)] = sparse_shape

        if with_non_object_images:
            to_ret.update({'images' : images})
        to_ret.update(indicators)
        for i in range(BATCH_SIZE):
            for k in to_ret:
                if ATTRIBUTE_SHAPES[k] is not None:
                    assert to_ret[k][i].shape == ATTRIBUTE_SHAPES[k], (k, to_ret[k][i].shape, ATTRIBUTE_SHAPES[k])
        return to_ret

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
                for i in range(batch_size):
                        datum = tf.train.Example(features = tf.train.Features(feature = {k : _bytes_feature(batch_data[k][i].tostring())}))
                        writer.write(datum.SerializeToString())

def write_in_thread((file_num, batches, write_path, prefix)):
    if prefix is None:
        prefix = file_num
    # Open writers 
    output_files = [os.path.join(write_path, attr_name, 
        str(prefix) + ':' + str(batches[0]) + ':' + str(batches[-1]) + '.tfrecords') for attr_name in ATTRIBUTE_NAMES]
    if KEEP_EXISTING_FILES:
        for i, output_file in enumerate(output_files):
            if os.path.isfile(output_file):
                print('Skipping file %s' % output_file)
                continue 
    writers = dict((attr_name, tf.python_io.TFRecordWriter(file_name)) \
            for (attr_name, file_name) in zip(ATTRIBUTE_NAMES, output_files))

    for _, batch in enumerate(batches):
        try:
            batch_data_dict = get_batch_data((file_num, batch), with_non_object_images = True)
        except ValueError as e:
            print('Error \'%s\' in batch %d - %d! Skipping batch' \
                    % (e, batches[0], batches[-1]))
            # Close writers
            for writer in writers.values():
                writer.close()
            for output_file in output_files:
                os.remove(output_file)
            return
        
        # TODO: Remove unneccessary data
        #batch_data_dict = remove_frames(batch_data_dict)
        # Write batch
        write_stuff(batch_data_dict, writers)
    # Close writers
    for writer in writers.values():
        writer.close()
    return 0

def do_write(all_images = True):
        if not os.path.exists(NEW_TFRECORD_TRAIN_LOC):
                os.mkdir(NEW_TFRECORD_TRAIN_LOC)
        if not os.path.exists(NEW_TFRECORD_VAL_LOC):
                os.mkdir(NEW_TFRECORD_VAL_LOC)

        for nm in ATTRIBUTE_NAMES:
                write_dir_train = os.path.join(NEW_TFRECORD_TRAIN_LOC, nm)
                write_dir_val = os.path.join(NEW_TFRECORD_VAL_LOC, nm)
                if not os.path.exists(write_dir_train):
                        os.mkdir(write_dir_train)
                if not os.path.exists(write_dir_val):
                        os.mkdir(write_dir_val)
	my_rng = np.random.RandomState(seed = 0)
        file_count = 0
	
	# create batch tasks
        write_tasks = []
        num_batches = 0
        for file_num in range(len(my_files)):
            write_task = []
            for batch_num in range(0, NUM_BATCHES, 4):
                if my_rng.rand() > 0.1:
                    write_path = NEW_TFRECORD_TRAIN_LOC
                else:
                    write_path = NEW_TFRECORD_VAL_LOC
                write_task.append((file_num, range(batch_num, batch_num + 4), write_path, PREFIX))
                num_batches += 1
            write_tasks.append(write_task)

        rs = []
        pools = []
        completed  = 0
        for write_task in write_tasks:
            for wt in write_task:
                write_in_thread(wt)
                completed += 1 
                print(str(float(completed) / num_batches * 100) + ' %')
        print('DONE')
        #for _ in tqdm(p.imap(write_in_thread, write_tasks), desc='batch', total=len(write_tasks)):
        #    pass

if __name__ == '__main__':
        do_write()
        for f in my_files:
                f.close()
