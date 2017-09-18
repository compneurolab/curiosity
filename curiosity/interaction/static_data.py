'''
A data provider module for when we are training offline.
Should keep the form of data.py
'''

import h5py
import threading
import numpy as np
import json
import six.moves.queue as queue
import cPickle

def check_obj_there(hdf5_filenames):
	hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
	print('starting check')
	for file_num, (src, filename) in enumerate(zip(hdf5s, hdf5_filenames)):
		msgs_all = src['msg']
		incomplete_filenum = False
		print(file_num)
		print(filename)
		for idx in range(0, msgs_all.shape[0], 32):
			msgs = msgs_all[idx : idx + 32]
			for msg in msgs:
				msg = json.loads(msg)
				if msg['msg']['action_type'] == 'OBJ_NOT_PRESENT':
					print((file_num, idx))
					incomplete_filenum = True
					break
			if incomplete_filenum:
				break

def print_some_actions(hdf5_filenames, batches_to_print = 2):
	hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
	print('printing some actions')
	for file_name, src in zip(hdf5_filenames, hdf5s):
		msgs_all = src['msg']
		for idx in range(0, batches_to_print * 32, 32):
			msgs = msgs_all[idx : idx + 32]
			for j, msg in enumerate(msgs):
				msg = json.loads(msg)
				print(str(idx + j) + str(msg['msg']))

def get_uniform_metadata(hdf5_filenames, save_loc, data_lengths, action_repeat_mod = 1):
	hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
	print('starting inspection')
	valid_idxs = []
	min_idx = max(data_lengths['obs']['depths1'], data_lengths['action'], data_lengths['action_post']) - 1
	for file_num, src in enumerate(hdf5s):
		msgs_all = src['msg']
		incomplete_filenum = False
		print(msgs_all.shape)
		for idx in range(min_idx, msgs_all.shape[0], 32):
			msgs = msgs_all[idx : idx + 32]
			for k, msg in enumerate(msgs):
				msg = json.loads(msg)
				if msg is not None and idx + k % action_repeat_mod == action_repeat_offset:
					valid_idxs.append((file_num, idx + k))
	print('num valid sequences: ' + str(len(valid_idxs)))
	metadata = {'filenames' : hdf5_filenames, 'valid_idxs' : valid_idxs}
	with open(save_loc, 'w') as stream:
		cPickle.dump(metadata, stream)
	return metadata



class UniformRandomBatcher:
	def __init__(self, batch_size, metadata, seed):
		print('initializing UniformRandomBatcher')
		self.batch_size = batch_size
		self.rng = np.random.RandomState(seed)
		self.valid_idxs = metadata['valid_idxs']
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


def get_objthere_metadata(hdf5_filenames, save_loc, data_lengths, action_repeat_mod = 1, action_repeat_offset = 0):
	min_idx = max(data_lengths['obs']['depths1'], data_lengths['action'], data_lengths['action_post']) - 1
	batch_size = 32
	obj_there_idxs = []
	obj_not_there_idxs = []
	hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
	for file_num, src in enumerate(hdf5s):
		msgs_all = src['msg']
		for idx in range(min_idx, msgs_all.shape[0], batch_size):
			msgs = msgs_all[idx : idx + batch_size]
			for k, msg in enumerate(msgs):
				msg = json.loads(msg)
				if msg is None or idx + k % action_repeat_mod != action_repeat_offset:
					continue
				if msg['msg']['action_type'] == 'OBJ_ACT':
					obj_there_idxs.append((file_num, idx + k))
				else:
					obj_not_there_idxs.append((file_num, idx + k))
	metadata = {'filenames' : hdf5_filenames, 'obj_there_idxs' : obj_there_idxs, 'obj_not_there_idxs' : obj_not_there_idxs}
	with open(save_loc, 'w') as stream:
		cPickle.dump(metadata, stream)
	return metadata

def get_objthere_metadata_deluxe(hdf5_filenames, save_loc, timesteps_before, timesteps_after, action_repeat_mod = 1, action_repeat_offset = 0):
	max_before = max(timesteps_before['obs']['depths1'], timesteps_before['action'], timesteps_before['action_post'])
	max_after = max(timesteps_after['obs']['depths1'], timesteps_after['action'], timesteps_after['action_post'])
	batch_size = 32
	obj_there_idxs = []
	obj_not_there_idxs = []
	hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
	for file_num, src in enumerate(hdf5s):
		msgs_all = src['msg']
		for idx in range(0, msgs_all.shape[0], batch_size):
			msgs = [json.loads(msg) for msg in msgs_all[idx : idx + batch_size]]
			for k, msg in enumerate(msgs):
				if msg is None or idx + k < max_before or idx + k + max_after >= msgs_all.shape[0] or (idx + k) % action_repeat_mod != action_repeat_offset:
					continue
				if msg['msg']['action_type'] == 'OBJ_ACT':
					obj_there_idxs.append((file_num, idx + k))
				else:
					obj_not_there_idxs.append((file_num, idx + k))
	metadata = {'filenames' : hdf5_filenames, 'obj_there_idxs' : obj_there_idxs, 'obj_not_there_idxs' : obj_not_there_idxs, 'timesteps_before' : timesteps_before, 'timesteps_after' : timesteps_after, 'action_repeat_mod' : action_repeat_mod, 'action_repeat_offset' : action_repeat_offset}
	with open(save_loc, 'w') as stream:
		cPickle.dump(metadata, stream)
	return metadata






class ObjectThereBatcher:
	def __init__(self, batch_size, metadata, seed, num_there_per_batch, num_not_there_per_batch):
		self.batch_size = batch_size
		self.rng = np.random.RandomState(seed)
		self.obj_there_idxs = metadata['obj_there_idxs']
		self.obj_not_there_idxs = metadata['obj_not_there_idxs']
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
		return list(obj_there_batch) + list(obj_not_there_batch)
		
		
class BetterOfflineDataProvider(threading.Thread):
	def __init__(self,
			batch_size,
			batcher_constructor,
			capacity,
			metadata_filename,
			batcher_kwargs = None):
		threading.Thread.__init__(self)
		with open(metadata_filename) as stream:
			metadata = cPickle.load(stream)
		hdf5_filenames = metadata['filenames']
		self.hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
		self.batcher = batcher_constructor(batch_size, metadata, ** batcher_kwargs)
		self.batch_size = batch_size
		self.t_before = metadata['timesteps_before']
		self.t_after = metadata['timesteps_after']
		self.queue = queue.Queue(capacity)
		self.daemon = True

	def run_env(self):
		while True:
			batch = {'recent' : {}}
                        chosen = self.batcher.get_batch_indices()
                        for k, v in self.t_before.iteritems():
                                if k == 'obs':
                                        for k_obs, v_obs in v.iteritems():
						t_plus = self.t_after['obs'][k_obs]
                                                collected_dat = []
                                                for (file_num, idx) in chosen:
                                                        dat_raw = self.hdf5s[file_num][k_obs][idx - v_obs : idx + t_plus + 1]
                                                        collected_dat.append(dat_raw)
                                                batch[k_obs] = np.array(collected_dat)
                                else:
					t_plus = self.t_after[k]
                                        collected_dat = []
                                        for file_num, idx in chosen:
                                                collected_dat.append(self.hdf5s[file_num][k][idx - v : idx + t_plus + 1])
                                        batch[k] = np.array(collected_dat)
#                       collected_dat = []
#                       for file_num, idx in chosen:
#                               collected_dat.append([self.hdf5s[file_num]['msg'][idx]])
#                       batch['recent']['msg'] = collected_dat
                        yield batch


        def start_runner(self, sess):
                self.start()

        def run(self):
                self._run()

        def _run(self):
                yielded = self.run_env()
                while True:
                        batch = next(yielded)
                        self.queue.put(batch, timeout = 5000.0)

        def dequeue_batch(self):
                return self.queue.get(timeout = 5000.0)

        def close(self):
                for f in self.hdf5s:
                        f.close()




class OfflineDataProvider(threading.Thread):
	def __init__(self, 
			batch_size, 
			batcher_constructor, 
			data_lengths, 
			capacity, 
			metadata_filename,
			batcher_kwargs = None):
		threading.Thread.__init__(self)
		with open(metadata_filename) as stream:
			metadata = cPickle.load(stream)
		hdf5_filenames = metadata['filenames']
		self.hdf5s = [h5py.File(fn, mode = 'r') for fn in hdf5_filenames]
		self.batcher = batcher_constructor(batch_size, metadata, ** batcher_kwargs)
		self.batch_size = batch_size
		self.data_lengths = data_lengths
		self.queue = queue.Queue(capacity)
		self.daemon = True

	def run_env(self):
		while True:
			batch = {'recent' : {}}
			chosen = self.batcher.get_batch_indices()
			for k, v in self.data_lengths.iteritems():
				if k == 'obs':
					for k_obs, v_obs in v.iteritems():
						collected_dat = []
						for (file_num, idx) in chosen:
							dat_raw = self.hdf5s[file_num][k_obs][idx - v_obs + 1 : idx + 1]
							collected_dat.append(dat_raw)
						batch[k_obs] = np.array(collected_dat)
				else:
					collected_dat = []
					for file_num, idx in chosen:
						collected_dat.append(self.hdf5s[file_num][k][idx - v + 1 : idx + 1])
					batch[k] = np.array(collected_dat)
#			collected_dat = []
#			for file_num, idx in chosen:
#				collected_dat.append([self.hdf5s[file_num]['msg'][idx]])
#			batch['recent']['msg'] = collected_dat
			yield batch

	def start_runner(self, sess):
		self.start()

	def run(self):
		self._run()

	def _run(self):
		yielded = self.run_env()
		while True:
			batch = next(yielded)
			self.queue.put(batch, timeout = 5000.0)

	def dequeue_batch(self):
		return self.queue.get(timeout = 5000.0)

	def close(self):
		for f in self.hdf5s:
			f.close()








