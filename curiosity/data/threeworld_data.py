import numpy as np
import json
import tensorflow as tf
import os

from tfutils.data import TFRecordsParallelByFileProvider

class ThreeWorldDataProvider(TFRecordsParallelByFileProvider):
    def __init__(self,
                 data_path,
                 sources,
                 n_threads=4,
                 batch_size=256,
                 delta_time=1,
                 sequence_len=1,
                 max_random_skip=None,
                 seed=0,
                 output_format='sequence', # 'pairs'
                 filters=None,
                 gaussian=None,
                 resize=None,
                 *args,
                 **kwargs):

        self.data_path = data_path
        self.sources = sources
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.delta_time = delta_time
        self.sequence_len = sequence_len
        self.max_skip = max_random_skip
        self.output_format = output_format
        self.filters = filters
        self.gaussian = gaussian
        self.seed = seed
        self.resize = resize
        
        assert self.delta_time >= 1,\
                ("delta time has to be at least 1")
        assert self.sequence_len >= 1,\
                ("sequence length has to be at least 1")
        assert self.batch_size >= self.sequence_len * self.delta_time,\
                ("batch size has to be at least equal to sequence length times \
                delta time")

        # total sequence length equals the requested sequence plus max_skip
        if self.max_skip is not None:
            assert self.max_skip >= 1,\
                ("max skip has to be at least 1")
            self.sequence_len += self.max_skip

        # load actions and positions from tfrecords for gaussian blob
        if self.gaussian is not None:
            if 'actions' not in sources:
                self.sources.append('actions')
            if 'object_data' not in sources:
                self.sources.append('object_data')
            has_image = False
            for image_data in ['images', 'normals', 'objects', \
                          'images2', 'normals2', 'objects2']:
                if image_data in self.sources:
                    has_image = True
                    break
            if not has_image:
                self.sources.append('images')

        # load sources from tfrecords
        self.source_paths = []
        for source in self.sources:
            self.source_paths.append(os.path.join(self.data_path, source))

        # load filters from tfrecords
        if self.filters is not None:
            for f in self.filters:
                self.source_paths.append(os.path.join(self.data_path, f))

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
        if self.resize is not None and source in self.resize:
            data = tf.image.convert_image_dtype(data, dtype=tf.float32)
            data = tf.image.resize_images(data,
                    self.resize[source], method=tf.image.ResizeMethod.BICUBIC)
            data = tf.image.convert_image_dtype(data, dtype=tf.uint8)
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
        init_filters = True
        for f in self.filters:
            if init_filters:
                all_filters = tf.ones(data[f].get_shape().as_list()[0:2], tf.int32)
                init_filters = False
            #TODO Nicer way: Take first element of filter (is_object_there 1st object)
            data[f] = tf.slice(data[f], [0,0,0], [-1,-1,1])
            data[f] = tf.cast(data[f], tf.int32)
            data[f] = tf.squeeze(data[f])
            # logical 'and' operation on all filters
            all_filters *= data[f]
            # remove filter from dict such that it is not enqueued
            data.pop(f)
        # check if ALL binary labels within sequence are not zero
        filter_sum = tf.reduce_sum(all_filters, 1)
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
        return data

    def random_skip(self, data):
        # get the current batch size
        batch_size = None
        for k in data:
            batch_size = tf.shape(data[k])[0]
            break
        if batch_size is None:
            raise ValueError('No batch size could be derived')

        # randomly skip after the requested sequence length to get the last frame
        rskip = tf.random_uniform([batch_size],\
                minval=1, maxval=self.max_skip, dtype=tf.int32, seed=self.seed)
        seq_len = self.sequence_len - self.max_skip
        for k in data:
            shape = data[k].get_shape().as_list()
            seq = tf.slice(data[k], [0]*len(shape), [-1,seq_len]+[-1]*(len(shape)-2))
            indices = tf.stack([tf.range(batch_size), self.sequence_len - rskip], axis=1)
            last_frame = tf.gather_nd(data[k], indices)
            last_frame = tf.expand_dims(last_frame, 1)
            data[k] = tf.concat([seq, last_frame], 1)
        data['random_skip'] = rskip
        return data

    def check_lengths(self, data):
        for k in data:
            if k is 'random_skip':
                assert len(data[k].get_shape().as_list()) == 1
            elif self.output_format is 'sequence':
                assert data[k].get_shape().as_list()[1] == self.sequence_len
            elif self.output_format is 'pairs':
                assert data[k].get_shape().as_list()[1] == 2

    def init_ops(self):
        self.input_ops = super(ThreeWorldDataProvider, self).init_ops()
        for i in range(len(self.input_ops)):
            if self.filters is not None:
                self.input_ops[i] = self.apply_filters(self.input_ops[i])
            if self.max_skip is not None:
                self.input_ops[i] = self.random_skip(self.input_ops[i])
        if self.max_skip is not None:
            self.sequence_len = self.sequence_len - self.max_skip + 1
        for i in range(len(self.input_ops)):
            self.check_lengths(self.input_ops[i])
        return self.input_ops
