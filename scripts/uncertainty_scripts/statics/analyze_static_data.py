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
HDF5_FILENAMES = [data_pfx + str(i) + '.hdf5' for i in range(1)]#due to seeding issue, the others should in fact be identical.
BATCH_SIZE = 32
T_PER_STATE = 2
NUM_TIMESTEPS = 3
UNIFORM_METADATA_LOC = '/media/data2/nhaber/uniform_metadata.pkl'
OBJTHERE_METADATA_LOC = '/media/data2/nhaber/objthere_metadata.pkl'

data_lengths = {
                        'obs' : {'depths1' : T_PER_STATE + NUM_TIMESTEPS},
                        'action' : T_PER_STATE - 1 + NUM_TIMESTEPS,
                        'action_post' : T_PER_STATE - 1 + NUM_TIMESTEPS}




#uniform_metadata = static_data.get_uniform_metadata(HDF5_FILENAMES, UNIFORM_METADATA_LOC, data_lengths)
#objthere_metadata = static_data.get_objthere_metadata(HDF5_FILENAMES, OBJTHERE_METADATA_LOC, data_lengths)

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











