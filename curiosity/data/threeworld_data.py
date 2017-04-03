import numpy as np
import json
import tensorflow as tf
import os

from tfutils.data import TFRecordsParallelByFileProvider

class ThreeWorldDataProvider(TFRecordsParallelByFileProvider):
    # image height and width
    image_height = 256
    image_width  = 256

    def __init__(self,
                 data_path,
                 sources,
                 n_threads=4,
                 batch_size=256,
                 delta_time=1,
                 sequence_len=1,
                 output_format='sequence', # 'pairs'
                 filters=None,
                 gaussian=False,
                 *args,
                 **kwargs):

        self.data_path = data_path
        self.sources = sources
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.delta_time = delta_time
        self.sequence_len = sequence_len
        self.output_format = output_format
        self.filters = filters
        self.gaussian = gaussian
        
        assert self.delta_time >= 1,\
                ("delta time has to be at least 1")
        assert self.sequence_len >= 1,\
                ("sequence length has to be at least 1")
        assert self.batch_size >= self.sequence_len * self.delta_time,\
                ("batch size has to be at least equal to sequence length times \
                delta time")

        # load sources from tfrecords
        self.source_paths = []
        for source in sources:
            self.source_paths.append(os.path.join(self.data_path, source))

        # load filters from tfrecords
        if self.filters is not None:
            for f in self.filters:
                self.source_paths.append(os.path.join(self.data_path, f))

        # load actions and positions from tfrecords for gaussian blob
        if self.gaussian is True:
            actions_path = os.path.join(self.data_path, 'actions')
            if actions_path not in self.source_paths:
                self.source_paths.append(actions_path)

            object_data_path = os.path.join(self.data_path, 'object_data')
            if object_data_path not in self.source_paths:
                self.source_paths.append(object_data_path)

        #TODO load ids from tfrecords
        # ids_path = os.path.join(self.data_path, 'ids')
        #if ids_path is not in self.source_paths:
        #    self.source_paths.append(ids_path)

        super(ThreeWorldDataProvider, self).__init__(
            self.source_paths,
            batch_size=batch_size,
            postprocess=self.postprocess(),
            n_threads=n_threads,
            *args, **kwargs)        

    def postprocess(self):
        pp_dict = {}
        #postprocess images
        for source in self.sources:
            pp_dict[source] = [(self.postprocess_to_sequence, ([source]), {})]
        if self.filters is not None:
            for f in self.filters:
                pp_dict[f] = [(self.postprocess_to_sequence, ([f]), {},)]
        return pp_dict

    def postprocess_to_sequence(self, data, source, *args, **kwargs):
        if data.dtype is tf.string:
            data = tf.decode_raw(data, self.meta_dict[source]['rawtype'])
            data = tf.reshape(data, [-1] + self.meta_dict[source]['rawshape'])
        data = self.set_data_shape(data)
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
        if self.output_format is 'sequence':
            data = self.create_sequence(data)
        elif self.output_format is 'pairs':
            data = self.create_pairs(data)
        else:
            raise ValueError('Unknown output format %s' % self.output_format)
        return data

    def create_pairs(self, data):
        data = tf.expand_dims(data, 1)
        shape = data.get_shape().as_list()
        shape[0] -= self.delta_time
        begin = [self.delta_time] + [0] * (len(shape) - 1)
        future = tf.slice(data, begin, shape)
        begin = [0] * len(shape)
        current = tf.slice(data, begin, shape)
        return tf.concat([current, future], 1)

    def create_sequence(self, data):
        data = tf.expand_dims(data, 1)
        shape = data.get_shape().as_list()
        shape[0] -= (self.sequence_len - 1) * self.delta_time
        shifts = []
        for i in range(self.sequence_len):
            begin = [i * self.delta_time] + [0] * (len(shape) - 1)
            shifts.append(tf.slice(data, begin, shape))
        return tf.concat(shifts, 1)

    def apply_filters(self, data):
        seq_len = tf.constant(self.sequence_len, dtype=tf.int32)
        for f in self.filters:
            data[f] = tf.cast(data[f], tf.int32)
            data[f] = tf.squeeze(data[f])
            # check if ALL binary labels within sequence are not zero
            filter_sum = tf.reduce_sum(data[f], 1)
            if self.output_format is 'sequence':
                pos_idx = tf.equal(filter_sum, seq_len)
            elif self.output_format is 'pairs':
                pos_idx = tf.equal(filter_sum, tf.constant(2, dtype=tf.int32))
            else:
                raise ValueError('Unknown output format')
            # gather positive examples for each data entry
            for k in data:
                shape = data[k].get_shape().as_list()
                shape[0] = -1
                data[k] = tf.gather(data[k], tf.where(pos_idx))
                data[k] = tf.reshape(data[k], shape)
            # remove filter from dict such that it is not enqueued
            data.pop(f)
        return data

    def init_ops(self):
        self.input_ops = super(ThreeWorldDataProvider, self).init_ops()
        for i in range(len(self.input_ops)):
            if self.filters is not None:
                self.input_ops[i] = self.apply_filters(self.input_ops[i]) 
        return self.input_ops
#TODO Add random skip
