'''
Alternate data provider, with some data gathered into short sequences, other data gathered into long sequences,
and a master filter (AND operation) passed on so losses can be masked.
'''



import numpy as np
import json
import tensorflow as tf
import os

from tfutils.data import TFRecordsParallelByFileProvider

class ShortLongSequenceDataProvider(TFRecordsParallelByFileProvider):
    def __init__(self,
                 data_path,
                 short_sources,
                 long_sources,
                 n_threads=4,
                 batch_size=256,
                 short_len=1,
                 long_len = 10,
                 min_len = None,
                 filters=None,
                 resize=None,
		 is_there_subsetting_rule = None,
		 is_in_view_subsetting_rule = None,
		 mirror = False,
		 uniform_adjustments = {},
                 *args,
                 **kwargs):

        self.data_path = data_path
        self.short_sources = short_sources
        self.long_sources = long_sources
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.short_len = short_len
        self.long_len = long_len
        self.min_len = min_len
        self.filters = filters
        self.resize = resize
	self.is_there_subsetting_rule = is_there_subsetting_rule
	self.mirror = mirror
	self.uniform_adjustments = uniform_adjustments
	self.is_in_view_subsetting_rule = is_in_view_subsetting_rule

	if filters is not None and 'is_object_there' in filters:
		assert is_there_subsetting_rule is not None
        
        assert self.short_len >= 1 and self.long_len >= 1,\
                ("sequence length has to be at least 1")
        assert self.batch_size >= self.long_len, \
                ("batch size has to be at least equal to sequence length")
        if min_len is None:
            self.min_len = self.short_len
        else:
            assert self.min_len >= self.short_len, \
                ('Min length must be at least short length')
	if is_in_view_subsetting_rule is not None:
		if is_in_view_subsetting_rule == 'last_seen_and_first_not':
			assert 'is_object_in_view' in filters and 'is_object_in_view2' in filters
		else:
			raise Exception('Not implemented yet!')

        self.source_paths = [os.path.join(self.data_path, source) for source in short_sources + long_sources]

        # load filters from tfrecords
        if self.filters is not None:
            for f in self.filters:
                self.source_paths.append(os.path.join(self.data_path, f))

        #TODO load ids from tfrecords
        # ids_path = os.path.join(self.data_path, 'ids')
        #if ids_path is not in self.source_paths:
        #    self.source_paths.append(ids_path)

        super(ShortLongSequenceDataProvider, self).__init__(
            self.source_paths,
            batch_size=batch_size,
            postprocess=self.postprocess(),
            n_threads=n_threads,
            *args, **kwargs)        

    def postprocess(self):
        pp_dict = {}
        #postprocess images
        for source in self.short_sources:
            pp_dict[source] = [(self.postprocess_to_sequence, ([source]), {})]
        for source in self.long_sources:
            pp_dict[source] = [(self.postprocess_to_sequence, ([source]), {})]
        if self.filters is not None:
            for f in self.filters:
                pp_dict[f] = [(self.postprocess_to_sequence, ([f]), {},)]
        return pp_dict

    def augmentation_postprocess(self, data, source):
	if self.mirror:
		raise Exception('Not yet implemented')
		data = self.mirror_postprocess(data)
	if self.uniform_adjustments:
		data = self.adjust_uniform(data, source)
	return data

    def adjust_uniform(self, data, source):
	if source in self.uniform_adjustments:
		adjustments = self.uniform_adjustments[source]
		for desc, op in [('brightness', tf.image.random_brightness), ('hue', tf.image.random_hue)]:
			if desc in adjustments:
				data = op(data, max_delta = adjustments[desc], seed = 0)
		for desc, op in [('contrast', tf.image.random_contrast), ('saturation', tf.image.random_saturation)]:
			if desc in adjustments:
				data = op(data, lower = adjustments[desc][0], upper = adjustments[desc][1], seed = 0)
	return data


    def postprocess_to_sequence(self, data, source, *args, **kwargs):
        if data.dtype is tf.string:
            data = tf.decode_raw(data, self.meta_dict[source]['rawtype'])
            data = tf.reshape(data, [-1] + self.meta_dict[source]['rawshape'])
        data = self.set_data_shape(data)
        if self.resize is not None and source in self.resize:
            data = tf.image.convert_image_dtype(data, dtype=tf.float32)
            data = tf.image.resize_images(data,
                    self.resize[source], method=tf.image.ResizeMethod.BICUBIC)
            data = tf.image.convert_image_dtype(data, dtype=tf.uint8)
	data = self.augmentation_postprocess(data, source)
        data = self.create_data_sequence(data, source)
        return data


    def set_data_shape(self, data):
        shape = data.get_shape().as_list()
        shape[0] = self.batch_size
        for s in shape:
            assert s is not None, ("Unknown shape", shape)
        data.set_shape(shape)
        return data

    def create_data_sequence(self, data, source):
        if source in self.short_sources:
            data = self.create_short_sequence(data)
        elif source in self.long_sources or source in self.filters:
            data = self.create_long_sequence(data)
        else:
            raise ValueError('Source not recognized in sequence creation: ' + source)
        return data

    def create_short_sequence(self, data):
        data = tf.expand_dims(data, 1)
        # shape = data.get_shape().as_list()
        # shape[0] -= self.min_len - 1
        shifts = [data[i : i + self.batch_size - (self.min_len - 1)] for i in range(self.short_len)]
        # shifts = [tf.slice()]
        return tf.concat(shifts, 1)

    def create_long_sequence(self, data):
        data = tf.expand_dims(data, 1)
        data_shape = data.get_shape().as_list()
        data_type = data.dtype
        data_augmented = tf.concat([data, tf.zeros([self.long_len - self.min_len] + data_shape[1:], dtype = data_type)], axis = 0)
        shifts = [data_augmented[i : i + self.batch_size - (self.min_len - 1)] for i in range(self.long_len)]
        return tf.concat(shifts, 1)

    def apply_filters(self, data):
        min_len = tf.constant(self.min_len, dtype = tf.int32)
        for f in self.filters:
            data[f] = tf.cast(data[f], tf.int32)
            data[f] = tf.squeeze(data[f])
	    print(f)
	    print(data[f])
	if 'is_object_there' in self.filters:
		if self.is_there_subsetting_rule == 'just_first':
			data['is_object_there'] = data['is_object_there'][:,:, 0]
			print(data['is_object_there'])
		else:
			raise Exception('Other types not implemented')
	if self.is_in_view_subsetting_rule is not None:
		if self.is_in_view_subsetting_rule == 'last_seen_and_first_not':
			raise Exception('Need to implement!')
		else:
			raise Exception('Other types not implemented!')
        #and operation
        prod_filters = data[self.filters[0]]
        for f in self.filters[1:]:
            prod_filters *= data[f]
        data['master_filter'] = prod_filters
        prod_filters_min_len = prod_filters[:, :min_len]
        filter_sum = tf.reduce_sum(prod_filters_min_len, 1)
        pos_idx = tf.equal(filter_sum, min_len)
        for k in data:
            shape = data[k].get_shape().as_list()
            shape[0] = -1
            data[k] = tf.gather(data[k], tf.where(pos_idx))
            data[k] = tf.reshape(data[k], shape)
        return data

    def check_lengths(self, data):
        for k in data:
            if k in self.short_sources:
                assert data[k].get_shape().as_list()[1] == self.short_len, k
            elif k in self.long_sources or k == 'master_filter' or k in self.filters:
                assert data[k].get_shape().as_list()[1] == self.long_len, (k, data[k].get_shape().as_list())

    def init_ops(self):
        self.input_ops = super(ShortLongSequenceDataProvider, self).init_ops()
        for i in range(len(self.input_ops)):
            if self.filters is not None:
                self.input_ops[i] = self.apply_filters(self.input_ops[i])
        for i in range(len(self.input_ops)):
            self.check_lengths(self.input_ops[i])
        return self.input_ops
