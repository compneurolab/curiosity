'''
A data provider module for when we are training offline.
Should keep the form of data.py
'''

import h5py
import threading
import numpy as np
import json

class UniformRandomBatcher:
	def __init__(self, hdf5s, batch_size, data_lengths, seed):
		print('initializing UniformRandomBatcher')
		self.batch_size = batch_size
		self.rng = np.random.RandomState(seed)
		min_idx = max(data_lengths['obs']['depths1'], data_lengths['action'], data_lengths['action_post']) - 1
		self.data_lengths = [len(src['depths1']) for src in hdf5s]
		self.valid_idxs = []
		for file_num, src in enumerate(hdf5s):
			msgs_all = src['msg']
			for idx in range(min_idx, msgs_all.shape[0], self.batch_size):
				msgs = msgs_all[idx : idx + self.batch_size]
				for msg in msgs:
					msg = json.loads(msg)
					if msg is not None:
						self.valid_idxs.append((file_num, idx))
		self.init_epoch()
		print('Initialization complete')


	def init_epoch(self):
		self.start = 0
		self.schedule = self.rng.permutation(self.valid_idxs)

	def get_batch_indices(self):
		if self.start + self.batch_size > len(self.schedule):
			self.init_epoch()
		retval = self.schedule[self.start : self.start + self.batch_size]
		self.start += self.batch_size
		return retval

class ObjectThereBatcher:
	def __init__(self, hdf5s, batch_size, data_lengths, seed, num_there_per_batch, num_not_there_per_batch):
		print('intializing ObjectThereBatcher')
		self.batch_size = batch_size
		self.rng = np.random.RandomState(seed)
		min_idx = max(data_lengths['obs']['depths1'], data_lengths['action'], data_lengths['action_post']) - 1
		self.obj_there_idxs = []
		self.obj_not_there_idxs = []
		for file_num, src in enumerate(hdf5s):
			msgs_all = src['msg']
			for idx in range(min_idx, msgs_all.shape[0], self.batch_size):
				msgs = msgs_all[idx : idx + self.batch_size]
				for msg in msgs:
					msg = json.loads(msg)
					if msg is None:
						continue
					if msg['msg']['action_type'] == 'OBJ_ACT':
						self.obj_there_idxs.append((file_num, idx))
					else:
						self.obj_not_there_idxs.append((file_num, idx))
		self.there_start = -1
		self.not_there_start = -1
		self.check_init()
		self.num_there_per_batch = num_there_per_batch
		self.num_not_there_per_batch = num_not_there_per_batch
		assert len(self.obj_there_idxs) > num_there_per_batch
		assert len(self.obj_not_there_idxs) > num_not_there_per_batch
		assert self.num_there_per_batch + self.num_not_there_per_batch == self.batch_size
		print('Initialization complete')


	def check_init(self):
		if self.there_start == -1 or self.there_start + self.num_there_per_batch > len(self.obj_there_idxs):
			self.there_start = 0
			self.there_schedule = self.rng.permutation(self.obj_there_idxs)
		if self.not_there_start == -1 or self.not_there_start + self.num_not_there_per_batch > len(self.obj_not_there_idxs):
			self.not_there_start = 0
			self.not_there_schedule = self.rng.permutation(self.obj_not_there_idxs)
	
	def get_batch_indices(self):
		self.check_init()
		obj_not_there_batch = self.not_there_schedule[self.not_there_start : self.not_there_start + self.num_not_there_per_batch]
		obj_there_batch = self.there_schedule[self.there_start : self.there_start + self.num_there_per_batch]
		self.there_start += self.num_there_per_batch
		self.not_there_start += self.num_not_there_per_batch
		return obj_there_batch + obj_not_there_batch
		
		


class OfflineDataProvider(threading.Thread):
	def __init__(self, 
			hdf5_filenames,
			batch_size, 
			batcher_constructor, 
			data_lengths, 
			capacity, 
			batcher_args = None):
		self.batcher = batcher
		self.hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
		self.batcher = batcher_constructor(self.hdf5s, batch_size, data_lengths, ** batcher_args)
		self.batch_size = batch_size
		self.data_lengths = data_lengths
		self.queue = queue.Queue(capacity)
		self.daemon = True

	def run_env(self):
		while True:
			chosen = self.batcher.get_batch_indices()
			for k, v in data_lengths.iteritems():
				if k == 'obs':
					for k_obs, v_obs in v.iteritems():
						collected_dat = []
						for (file_num, idx) in chosen:
							dat_raw = self.hdf5s[file_num][k_obs][idx - v_obs + 1 : idx + 1]
							collected_dat.append(dat_raw)
						collected_dat.append(nones_replaced)
					batch[k_obs] = np.array(collected_dat)
				else:
					collected_dat = []
					for file_num, idx in chosen:
						collected_dat.append(self.hdf5s[file_num][k][idx - v + 1 : idx + 1])
					batch[k] = np.array(collected_dat)
			collected_dat = []
			for file_num, idx in chosen:
				collected_dat.append(self.hdf5s[file_num]['msg'][idx])
			batch['recent']['msg'] = collected_dat
			yield batch

	def start_runner(self, sess):
		self.start()

	def run(self):
		self._run()

	def _run(self):
		yielded = self.run_env()
		while True:
			history = next(yielded)
			batch = self.batching_fn(history)
			self.queue.put(batch, timeout = 5000.0)

	def dequeue_batch(self):
		return self.queue.get(timeout = 5000.0)










