'''
An explicit position data provider.

Meant to deal with positions_parsed data written in explicit_pos_writer.py in scripts.
'''

from tfutils.data import TFRecordsParallelByFileProvider
import os
import tensorflow as tf


class PositionPredictionData(TFRecordsParallelByFileProvider):
	def __init__(self, 
				data_path, 
				batch_size = 256, 
				n_threads = 1, 
				num_timesteps = 6, 
				max_num_objects = 100, 
				max_num_actions = 10, 
				time_skip = 1, 
				positions_only = False, 
				output_num_objects = None, 
				output_num_actions = None, 
				remove_teleport = True,
				filters = None,
				* args, 
				** kwargs):
		self.filters = filters
		self.batch_size = batch_size
		self.source_paths = [os.path.join(data_path, 'positions')]
		self.num_timesteps = num_timesteps
		self.time_skip = time_skip
		self.max_num_objects = max_num_objects
		self.max_num_actions = max_num_actions
		self.positions_only = positions_only
		self.output_num_objects = max_num_objects if output_num_objects is None else output_num_objects
		self.output_num_actions = max_num_actions if output_num_actions is None else output_num_actions
		self.remove_teleport = remove_teleport
		if time_skip != 1:
			raise NotImplementedError('Time skip not yet implemented')
		if filters is not None:
			for f in self.filters:
				self.source_paths.append(os.path.join(data_path, f))
		super(PositionPredictionData, self).__init__(
			self.source_paths, 
			postprocess = {'positions': [(self.postprocess_positions, (), {})]}, 
			batch_size = batch_size, 
			n_threads = n_threads, * args, **kwargs)

	def postprocess_positions(self, positions):
		positions = tf.decode_raw(positions, tf.float32)
		pos_shape = [self.batch_size, 3 * self.max_num_objects + 3 * self.max_num_actions]
		positions.set_shape(pos_shape)
		return positions

	def toss_and_cut(self, data):
		if self.remove_teleport:
			data['positions'] = data['positions'][1:]
		if not self.positions_only:
			data['corresponding_actions'] = data['positions'][:, 3 * self.max_num_objects : 3 * (self.max_num_objects + self.output_num_actions)]
		data['positions'] = data['positions'][:, : 3 * self.output_num_objects]

		return data

	def create_sequence(self, data):
		num_to_append = data['positions'].get_shape().as_list()[0] - self.num_timesteps
		shifts = []
		for i in range(self.num_timesteps):
			shifts.append(data['positions'][i:i + num_to_append])
		data['positions'] = tf.concat(1, shifts)

		if not self.positions_only:
			num_to_append = data['corresponding_actions'].get_shape().as_list()[0] - self.num_timesteps
			shifts = []
			for i in range(self.num_timesteps):
				shifts.append(data['corresponding_actions'][i:i + num_to_append])
			data['corresponding_actions'] = tf.concat(1, shifts)

		return data

	def init_ops(self):
		self.input_ops = super(PositionPredictionData, self).init_ops()

		for i in range(len(self.input_ops)):
			self.input_ops[i] = self.toss_and_cut(self.input_ops[i]) #might need to change ordering here, or think about this...
			self.input_ops[i] = self.create_sequence(self.input_ops[i])

			if self.filters is not None:
				self.input_ops[i] = self.apply_filters(self.input_ops[i])

		return self.input_ops

	def apply_filters(self, data):
		delta_t = tf.constant(self.num_timesteps, self.int32)

		for f in self.filters:
			data[f] = tf.decode_raw(data[f], tf.int32)
			shape = data[f].get_shape().as_list()

			#making sure tensors are 2d
			if len(shape) < 2:
				data[f] = tf.expand_dims(data[f], -1)
				shape = data[f].get_shape().as_list()
			shape[1] = 1
			data[f].set_shape(shape)

			#toss and cut analogue...should modify toss and cut
			if self.remove_teleport:
				data[f] = data[f][1:]

			#create sequence analog...should modify create_sequence
			num_to_append = data[f].get_shape().as_list()[0] - self.num_timesteps
			shifts = []
			for i in range(self.num_timesteps):
				shifts.append(data[f][i:i + num_to_append])
			data[f] = tf.concat(1, shifts)

			filter_sum = tf.reduce_sum(data[f], 1)
			pos_idx = tf.equal(filter_sum, delta_time)
			data.pop(f)
			for k in data:
				shape = data[k].get_shape().as_list()
				shape[0] = -1
				data[k] = tf.gather(data[k], tf.where(pos_idx))
				data[k] = tf.reshape(data[k], shape)

		return data











