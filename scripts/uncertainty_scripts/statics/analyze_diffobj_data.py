'''
Script for analyzing static data.

No training, just seeing if everything is ok, checking read speed, etc.
'''





import sys
sys.path.append('curiosity')
sys.path.append('tfutils')
import tensorflow as tf
import time

from curiosity.interaction import train, environment, static_data, cfg_generation, update_step
import curiosity.interaction.models as models
from tfutils import base, optimizer
import numpy as np


data_pfx = '/media/data2/nhaber/offline_data'
#HDF5_FILENAMES = [data_pfx + str(i) + '.hdf5' for i in range(1)]#due to seeding issue, the others should in fact be identical.
force_scaling = 200
#TRAIN_HDF5_FILENAMES = ['/media/data2/nhaber/rel' + str(int(force_scaling)) + '_' + str(tasknum) + '.hdf5' for tasknum in [4, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1000, 1001, 1002, 1010, 1011, 1012, 1016, 1018, 1020, 1021]
#if tasknum not in [30, 31, 33, 34, 36, 38, 41, 43, 44, 48, 1001, 1002] and tasknum not in [32, 37, 39, 42, 46, 49, 1010, 1011, 1012, 1016, 1018, 1020, 1021]]#first exception has not-theres, second exception had error
#41 is odd, apparently object is there but is never seen
#TEST_HDF5_FILENAMES = ['/media/data2/nhaber/rel' + str(int(force_scaling)) + '_' + str(tasknum) + '.hdf5' for tasknum in [2000, 2001, 2002, 2004, 2005, 2007, 2050, 2051, 2052, 2053, 2054, 2057]]


#SMALLER_TEST_FILENAMES = ['/media/data2/nhaber/rel' + str(int(force_scaling)) + '_' + str(tasknum) + '.hdf5' for tasknum in [2000, 2001, 2002]]
the_chosen_ones = range(1, 13) + [14, 17, 19, 22]
SMALLER_TEST_FILENAMES = ['/media/data4/nhaber/one_room_dataset/bigroom/val_diffobj' + str(i) + '.hdf5' for i in the_chosen_ones]
SMALLER_META_FILENAMES = ['/media/data4/nhaber/one_room_dataset/bigroom/val_diffobj' + str(i) + '_meta.pkl' for i in the_chosen_ones]
JOINED_META_FILENAME = '/media/data4/nhaber/one_room_dataset/bigroom/val_diffobj_all_meta.pkl'

BATCH_SIZE = 32
#T_PER_STATE = 2
#NUM_TIMESTEPS = 1
#TRAIN_UNIFORM_METADATA_LOC = '/media/data2/nhaber/train_ts3_uniform_rel200.pkl'
#TRAIN_OBJTHERE_METADATA_LOC = '/media/data2/nhaber/train_ts3_objthere_rel200.pkl'
#TEST_UNIFORM_METADATA_LOC = '/media/data2/nhaber/test_ts3_uniform_rel200.pkl'
#TEST_OBJTHERE_METADATA_LOC = '/media/data4/nhaber/one_room_dataset/diffobj_try1.pkl'
state_desc = 'images1'


data_lengths = {
                        'obs' : {state_desc : 7},
                        'action' : 6,
                        'action_post' : 6}




#static_data.check_obj_there(TRAIN_HDF5_FILENAMES)
#good_until = static_data.check_obj_there(SMALLER_TEST_FILENAMES)
print('done checking')

#train_uniform_metadata = static_data.get_uniform_metadata(TRAIN_HDF5_FILENAMES, TRAIN_UNIFORM_METADATA_LOC, data_lengths)
#train_objthere_metadata = static_data.get_objthere_metadata(TRAIN_HDF5_FILENAMES, TRAIN_OBJTHERE_METADATA_LOC, data_lengths)
#test_uniform_metadata = static_data.get_uniform_metadata(TEST_HDF5_FILENAMES, TEST_UNIFORM_METADATA_LOC, data_lengths)
#test_objthere_metadatas = [static_data.get_objthere_metadata([fn], ml, data_lengths, state_desc = state_desc) for fn, ml in zip(SMALLER_TEST_FILENAMES, SMALLER_META_FILENAMES)]
all_meta = static_data.get_objthere_metadata(SMALLER_TEST_FILENAMES, JOINED_META_FILENAME, data_lengths, state_desc = state_desc)

#static_data.save_some_objthere_images(test_objthere_metadatas, '/media/data4/nhaber/one_room_dataset/try_wimg', how_many = 10, state_desc = state_desc)
'''
dummy_sess = None
#rand_dp = static_data.OfflineDataProvider(BATCH_SIZE, static_data.UniformRandomBatcher, data_lengths, 5, UNIFORM_METADATA_LOC, batcher_kwargs = {'seed' : 0})
#rand_dp.start_runner(dummy_sess)
#last_time = time.time()
#for i in range(10):
#	batch = rand_dp.dequeue_batch()
#	curr_time = time.time()
#	print(curr_time - last_time)
#	last_time = curr_time
#	print(batch.keys())
#	print(batch['depths1'].shape)
#	print(batch['action'].shape)
#	print(batch['action_post'].shape)
#	print(batch['action'][0])
#	print(batch['action_post'][0])

 
balancing_dp = static_data.OfflineDataProvider(BATCH_SIZE, static_data.ObjectThereBatcher, data_lengths, 5, OBJTHERE_METADATA_LOC, batcher_kwargs = {'seed' : 0, 'num_there_per_batch' : 16, 'num_not_there_per_batch' : 16})
balancing_dp.start_runner(dummy_sess)
last_time = time.time()
for i in range(10):
	batch = balancing_dp.dequeue_batch()
	curr_time = time.time()
	print(curr_time - last_time)
	last_time = curr_time
	print(batch['depths1'].shape)
#	print(batch['action'].shape)
#	print(batch['action_post'].shape)
#	print(batch['action'][:,-1])
#	print(batch['action_post'][:, -1])

'''









