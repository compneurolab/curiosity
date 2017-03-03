'''
Script for writing new tfrecord, a mask for data inclusion based on what sort of action is taking place.
'''


import h5py
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import json

val_time = True

if val_time:
    print('writing validation')
else:
    print('writing training')

data_dir = '/media/data/one_world_dataset'

if val_time:
    data_file = os.path.join(data_dir, 'dataset8.hdf5')
    data2_dir = '/media/data2/one_world_dataset/tfvaldata'
else:
    data_file = os.path.join(data_dir, 'dataset.hdf5')
    data2_dir = '/media/data2/one_world_dataset/tfdata'


f = h5py.File(data_file, 'r')





def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def resize_img(image):
    return np.array(Image.fromarray(image).resize((NEW_HEIGHT, NEW_WIDTH), Image.BICUBIC))

def mean_img(image):
    return np.mean(np.array(image).astype(np.float))

def get_world_info(my_range):
    return [json.loads(f['worldinfo'][i]) for i in my_range]

def get_actions(my_range):
    return [json.loads(f['actions'][i]) for i in my_range]

def screen_out_statics(observed_objects):
    return [obj for obj in observed_objects if (not obj[-1]) or obj[1] == -1]

def get_objects_acted_on(relevant_actions, persistent_data = None):
    relevant_ids= set([int(act_instance['id']) for act in relevant_actions for act_instance in act['actions']])
    if persistent_data is None:
        return list(relevant_ids)
    else:
        return list(set.intersection(relevant_ids, persistent_data))

def get_positions(world_info, persistent_ids = None):
    if persistent_ids is None:
        return [dict((obj_info[1], np.array(obj_info[2])) for obj_info in info['observed_objects']) for info in world_info]
    else:
        return [dict((obj_info[1], np.array(obj_info[2])) for obj_info in info['observed_objects'] if obj_info[1] in persistent_ids) for info in world_info]

    
    
def get_second_frame_distances(positions, relative_to):
    first_positions = positions[1]
    return [(idx, np.linalg.norm(pos - relative_to)) for (idx, pos) in first_positions.iteritems()]

def get_second_frame_closeness_sorting(positions, relative_to):
    first_distances = get_second_frame_distances(positions, relative_to)
    return sorted(first_distances, key = lambda (a, b) : b)

def get_persistent_objects(batch_world_info):
    objects = [screen_out_statics(frame_ex['observed_objects']) for frame_ex in batch_world_info]
    positions = [dict((obj[1], obj[2])for obj in fr_objs) for fr_objs in objects]
    obj_ids = [set([obj[1] for obj in fr_objs]) for fr_objs in objects]
    persistent_objects = set.intersection(* obj_ids)
    return persistent_objects

def get_positions_arr(objects_of_interest, positions, n_objects):
    positions_list = []
    num_gotten = len([obj for obj in objects_of_interest if obj != -1]) + 1
    if num_gotten > n_objects:
        objects_of_interest = objects_of_interest[:-1]
    for frame_positions in positions:
        agent_position = frame_positions[-1]
        frame_pos_list = [agent_position] + [frame_positions[i] for i in objects_of_interest if i != -1]
        frame_pos_arr = np.concatenate(frame_pos_list)
        frame_pos_arr = np.reshape(frame_pos_arr, (1, len(frame_pos_arr)))
        positions_list.append(frame_pos_arr)
    retval = np.concatenate(positions_list)
    if n_objects > num_gotten:
        retval = np.concatenate([retval, np.zeros((len(positions), 3 * (n_objects - num_gotten)))], 1)
    return retval

def reformat_actions(frame_actions):
    return dict((int(act['id']), (act['force'], act['torque'])) for act in frame_actions['actions'])
        

def get_actions_arr(acted_on, batch_actions, max_acted_on):
    if len(acted_on) == 0:
        return np.zeros((len(batch_actions), 3 * max_acted_on))
    acted_on = acted_on[:max_acted_on]
    reformatted_actions = [reformat_actions(frame_actions) for frame_actions in batch_actions]
    action_list = []
    for frame_reformatted_actions in reformatted_actions:
        frame_act_list = []
        for obj_id in acted_on:
            if obj_id in frame_reformatted_actions:
                f, t = frame_reformatted_actions[obj_id]
                frame_act_list.append(np.array(f))
            else:
                frame_act_list.append(np.zeros(3))
        frame_act_arr = np.concatenate(frame_act_list)
        action_list.append(frame_act_arr.reshape(1, len(frame_act_arr)))
    retval = np.concatenate(action_list)
    if max_acted_on > len(acted_on):
        retval = np.concatenate([retval, np.zeros((len(batch_actions), 3 * (max_acted_on - len(acted_on))))], 1)
    return retval
    
    
    
    
def transform_batch(current_range, max_num_objects, max_acted_on):
    batch_world_info = get_world_info(current_range)
    batch_actions = get_actions(current_range)
    persistent_objects = get_persistent_objects(batch_world_info)
    acted_on = get_objects_acted_on(batch_actions, persistent_data = persistent_objects)
    positions = get_positions(batch_world_info, persistent_ids = persistent_objects)
    close_rank = get_second_frame_closeness_sorting(positions, positions[1][-1])
    close_no_actions = [idx for (idx, dist) in close_rank if idx not in acted_on]    
    objects_of_interest = list(acted_on) + close_no_actions
    objects_of_interest = objects_of_interest[:max_num_objects]
    positions_arr = get_positions_arr(objects_of_interest, positions, max_num_objects).astype('float32')
    actions_arr = get_actions_arr(acted_on, batch_actions, max_acted_on).astype('float32')
    return np.concatenate([positions_arr, actions_arr], 1)


if __name__ == '__main__':


        if val_time:
            N = 128000
        else:
            N = 2048000

        BATCH_SIZE = 256.0
        SCREEN_WIDTH = 512
        SCREEN_HEIGHT = 384
        NEW_WIDTH = 256
        NEW_HEIGHT = 256
        CHANNELS = 3
        max_num_objects = 100
        max_acted_on = 10

        write_dir = os.path.join(os.path.join(data2_dir, 'act_mask_1'))
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

        file_count = 0
        # batches = [4] * 2000
        # print('all_batches', sum(batches))
        batch_count = 0

        outputs = ['act_mask_1']

        # output_files = []
        # for output in outputs:
        #     output_files.append(os.path.join(data2_dir, output, str(k) + '.tfrecords'))
        # writers = []
        # for output_file in output_files:
        #     writers.append(tf.python_io.TFRecordWriter(output_file))

        for frame_num in range(N):
            if frame_num % 256 == 0:
                batch_count = int(frame_num / 256)
                current_range = range(256 * batch_count, 256 * (batch_count + 1))
                # batch_data = transform_batch(current_range, max_num_objects, max_acted_on)
            if frame_num % 1024 == 0:
                file_count = int(frame_num / 1024) 
                print('file count: ' + str(file_count))
                output_files = []
                for output in outputs:
                    output_files.append(os.path.join(data2_dir, output, str(file_count) + '.tfrecords'))
                writers = []
                for output_file in output_files:
                    writers.append(tf.python_io.TFRecordWriter(output_file))

            action_type = json.loads(f['actions'][frame_num])['action_type']
            if action_type not in ['TELEPORT', 'LOOK', 'MOVING_CLOSER']:
                data = [np.array([1]).astype(np.int32).tostring()]
            else:
                data = [np.array([0]).astype(np.int32).tostring()]



            # img = resize_img(f['images'][t])
            # #mea = mean_img(img)
            # #hgt = img.shape[0]
            # #wdt = img.shape[1]
            # #cha = img.shape[2]
            # data[0] = img = img.tostring()
            # data[1] = nor = resize_img(f['normals'][t]).tostring()
            # data[2] = obj = resize_img(f['objects'][t]).tostring()
            # data[3] = inf = f['worldinfo'][t]
            # data[4] = act = f['actions'][t]
            # data[5] = par = parse_action(act, [SCREEN_HEIGHT, SCREEN_WIDTH], \
            #                                   [NEW_HEIGHT, NEW_WIDTH]).tostring()
            #val = f['valid'][t]
            #idx = t
        #    datum = tf.train.Example(features=tf.train.Features(feature={
        #           'images': _bytes_feature(img),
        #           'normals': _bytes_feature(nor),
        #           'objects': _bytes_feature(obj),
        #           'height': _int64_feature(hgt),
        #           'width': _int64_feature(wdt),
        #           'channels': _int64_feature(cha),
        #           'worldinfo': _bytes_feature(inf),
        #           'actions': _bytes_feature(act),
        #          'parsed_actions': _bytes_feature(par),
        #           'ids': _int64_feature(int(idx)),
        #           'means': _float_feature(float(mea)),
        #       }))

            for i in range(len(outputs)):
                datum = tf.train.Example(features=tf.train.Features(feature={
                        outputs[i]: _bytes_feature(data[i]),
                }))
                writers[i].write(datum.SerializeToString())

        for writer in writers:
            writer.close()
        f.close()
