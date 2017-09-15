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
TRAIN_HDF5_FILENAMES = ['/media/data2/nhaber/rel' + str(int(force_scaling)) + '_' + str(tasknum) + '.hdf5' for tasknum in [0, 1, 4, 20, 21]]
TEST_HDF5_FILENAMES = ['/media/data2/nhaber/rel' + str(int(force_scaling)) + '_' + str(tasknum) + '.hdf5' for tasknum in [22]]
BATCH_SIZE = 32
#T_PER_STATE = 2
#NUM_TIMESTEPS = 1
TRAIN_UNIFORM_METADATA_LOC = '/media/data2/nhaber/train_uniform_rel200.pkl'
TRAIN_OBJTHERE_METADATA_LOC = '/media/data2/nhaber/train_objthere_rel200.pkl'
TEST_UNIFORM_METADATA_LOC = '/media/data2/nhaber/test_uniform_rel200.pkl'
TEST_OBJTHERE_METADATA_LOC = '/media/data2/nhaber/test_objthere_rel200.pkl'

data_lengths = {
                        'obs' : {'depths1' : 5},
                        'action' : 4,
                        'action_post' : 4}




print('checking if object there')
static_data.check_obj_there(TRAIN_HDF5_FILENAMES)
static_data.check_obj_there(TEST_HDF5_FILENAMES)
print('done checking')

train_uniform_metadata = static_data.get_uniform_metadata(TRAIN_HDF5_FILENAMES, TRAIN_UNIFORM_METADATA_LOC, data_lengths)
train_objthere_metadata = static_data.get_objthere_metadata(TRAIN_HDF5_FILENAMES, TRAIN_OBJTHERE_METADATA_LOC, data_lengths)
test_uniform_metadata = static_data.get_uniform_metadata(TEST_HDF5_FILENAMES, TEST_UNIFORM_METADATA_LOC, data_lengths)
test_objthere_metadata = static_data.get_objthere_metadata(TEST_HDF5_FILENAMES, TEST_OBJTHERE_METADATA_LOC, data_lengths)

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









