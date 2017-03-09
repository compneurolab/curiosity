'''
Script for writing new tfrecord.
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


baddies = ['SEARCHING', 'ROTATING', 'STANDUP', 'TELEPORT', 'MOVING_CLOSER', 'NO_CRASH', 'LOOK']
good_actions = ['MOVING_OBJECT', 'LIFTING', 'CRASHING']
keep_last = ['WAITING']





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
    
def get_obj_of_interest(actions):
    of_interest_now = None
    retval = []
    for act in actions:
        act_type = str(act['action_type'])
        if act_type in baddies:
            of_interest_now = None
        elif act_type in good_actions:
            of_interest_now = int(act['actions'][0]['id'])
        retval.append(of_interest_now)
    return retval

def get_positions_actions_of_interest(positions, actions, objects_of_interest):
    reform_actions = [reformat_actions(frame_actions) for frame_actions in actions]
    pos_list = []
    for (i, (obj, pos, ref_act)) in enumerate(zip(objects_of_interest, positions, reform_actions)):
        inst_list = []
        if obj is None or obj == 0 or obj not in pos:
            inst_list.extend([0.] * 6)
            objects_of_interest[i] = None
        else:
            inst_list.extend(pos[obj])
            if obj in ref_act:
                f, t = ref_act[obj]
                inst_list.extend(f)
            else:
                inst_list.extend([0., 0., 0.])
        pos_list.append(inst_list)
    return np.array(pos_list, dtype = 'float32')


def transform_batch(current_range):
    actions = get_actions(current_range)
    info = get_world_info(current_range)
    positions = get_positions(info)
    obj_of_interest = get_obj_of_interest(actions)
    return obj_of_interest, get_positions_actions_of_interest(positions, actions, obj_of_interest)


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

        file_count = 0
        # batches = [4] * 2000
        # print('all_batches', sum(batches))
        batch_count = 0

        outputs = ['pos_squeezed', 'mask_squeezed']

        for nm in outputs:
            write_dir = os.path.join(os.path.join(data2_dir, nm))
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)


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
                ooi, pos_act = transform_batch(current_range)
            if frame_num % 1024 == 0:
                file_count = int(frame_num / 1024) 
                print('file count: ' + str(file_count))
                output_files = []
                for output in outputs:
                    output_files.append(os.path.join(data2_dir, output, str(file_count) + '.tfrecords'))
                writers = []
                for output_file in output_files:
                    writers.append(tf.python_io.TFRecordWriter(output_file))

            data = [pos_act[frame_num % 256].tostring()]
            if ooi[frame_num % 256] is None:
                data.append(np.array([0]).astype(np.int32).tostring())
            else:
                data.append(np.array([1]).astype(np.int32).tostring())



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
