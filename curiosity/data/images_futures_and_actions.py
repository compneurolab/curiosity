from tfutils.data import TFRecordsParallelByFileProvider
import numpy as np
import json
from scipy.misc import imresize
import tensorflow as tf
import copy
import os
import cPickle

class FuturePredictionData(TFRecordsParallelByFileProvider):
    # counts the examples produced by this data provider
    example_counter = tf.constant(0)
    # epsilon used during normalization to avoid division by zero
    epsilon = np.array(1e-4).astype(np.float32)

    # image height and width
    image_height = 256
    image_width  = 256

    def __init__(self,
                 data_path,
                 batch_size=256,
                 crop_size=None,
                 min_time_difference=1, # including, also specifies fixed time
                 output_format={'images': 'pairs', 'actions': 'sequence'},
                 filters=None,
                 use_object_ids=True,
                 normalize_actions=None,
                 normalize_images=None,
                 stats_file=None,
                 action_matrix_radius=None,
                 is_remove_teleport=True,
                 use_poses=False,
                 n_threads=4,
                 *args,
                 **kwargs):

        self.orig_shape = None
        self.output_format = output_format
        self.use_object_ids = use_object_ids
        self.normalize_actions = normalize_actions
        self.normalize_images = normalize_images
        self.stats_file = stats_file
        self.is_remove_teleport = is_remove_teleport
        self.action_matrix_radius = action_matrix_radius
        self.filters = filters
        self.use_poses = use_poses

        self.images = 'images'
        self.actions = 'parsed_actions' #'actions'
        if self.use_poses:
            self.actions = 'poses' #'actions'

        if int(min_time_difference) < 1:
   	    self.min_time_difference = 1
	    print("The minimum time difference has to be at least 1, " \
	        + "and thus was set to 1.") 
        else:
	    self.min_time_difference = int(min_time_difference)

        self.batch_size = batch_size
        if self.batch_size <= self.min_time_difference:
            raise IndexError('batch_size has to be bigger than min_time_difference!')

        # load dataset statistics from provided file if necessary
        if self.normalize_actions is not None or self.normalize_images is not None:
            if self.stats_file is None:
                raise IndexError('If you want to normalize, you need to provide \
                                  a dataset statistics .pkl file')
            with open(self.stats_file) as f:
                self.stats = cPickle.load(f)

            # make sure everything is float32
            for k in ['mean', 'std', 'min', 'max']:
                self.stats[k][self.actions] = \
                    self.stats[k][self.actions].astype(np.float32)
            # make sure everything is float32 and resize means to [256, 256, 3]
            for k in ['mean', 'std']:
                self.stats[k][self.images] = \
                    self.stats[k][self.images].astype(np.float32)
                self.stats[k][self.images] = \
                    imresize(self.stats[k][self.images], \
                        [self.image_height, self.image_width, 3]).astype(np.float32)

            # correct zero entries in action_max to prevent nan loss
            self.stats['max'][self.actions][self.stats['max'][self.actions] == 0] = 1.0

        # load images and parsed actions from tfrecords
        self.source_paths = [os.path.join(data_path, self.images),
                             os.path.join(data_path, self.actions)]

        # also load filters
        if self.filters is not None:
            for f in self.filters:
                self.source_paths.append(os.path.join(data_path, f))

        super(FuturePredictionData, self).__init__(
            self.source_paths,
            batch_size=batch_size,
            postprocess={self.images: [(self.postproc_img, (), {})],
                        self.actions: [(self.postproc_parsed_actions, (), {})] },
                        #self.postproc_actions},
            n_threads=n_threads,
            *args, **kwargs)

        self.crop_size = crop_size
        if self.crop_size is not None:
            raise NotImplementedError('Do not use crop_size as it is not implemented!')

    def postproc_img(self, images):
        #TODO Implement cropping via warping
        if self.normalize_images is 'standard':
            images = tf.cast(images, tf.float32)
            images = (images - self.stats['mean'][self.images]) / \
                     (self.stats['std'][self.images] + self.epsilon)
        elif self.normalize_images is not None:
            raise TypeError('Unknown normalization type for images')

        return images

    def postproc_parsed_actions(self, actions):
        if self.use_poses:
            actions = tf.decode_raw(actions, tf.float32)
            actions = tf.squeeze(actions)
            actions = tf.slice(actions, [0,0], [-1, 4])
            act_shape = actions.get_shape().as_list()
            act_shape[1] = 4
            actions.set_shape(act_shape)
        return actions

        actions = tf.decode_raw(actions, tf.float64)
        actions = tf.cast(actions, tf.float32)

        act_shape = actions.get_shape().as_list()
        act_shape[1] = 25
        actions.set_shape(act_shape)

        #normalize actions
        if self.normalize_actions is 'standard':
            actions = (actions - self.stats['mean'][self.actions]) / \
                      (self.stats['std'][self.actions] + self.epsilon)
        elif self.normalize_actions is 'minmax':
            actions = (actions - self.stats['min'][self.actions]) / \
                      (self.stats['max'][self.actions] - self.stats['min'][self.actions])
        elif self.normalize_actions is 'custom':
            features = []
            for i in range(25):
                features.append(tf.slice(actions, [0, i], [-1, 1]))
            #clip at 0.28
            features[5] = tf.maximum(tf.minimum(features[5], 0.28), -0.28)
            #minmax norm on features
            for i in [2, 3, 5, 7, 8, 9, 11, 16, 18, 20]:
                features[i] = (features[i] - self.stats['custom_min'][self.actions][i]) \
                     / (self.stats['custom_max'][self.actions][i] - \
                     self.stats['custom_min'][self.actions][i])
            #clip at 10 and then sigmoid
            for i in [4, 6]:
                features[i] = tf.sigmoid(tf.maximum(tf.minimum(features[i], 10), -10))
            #divide by 255
            for i in [14, 15, 23, 24]:
                features[i] = tf.divide(features[i], 255)
            #toss entries
            actions = []
            for i in range(25):
                if i not in [0, 1, 10, 12, 13, 17, 19, 21, 22]:
                    actions.append(features[i])
            actions = tf.concat(actions, 1)

        elif self.normalize_actions is 'toss_zeros':
            features = []
            for i in range(25):
                features.append(tf.slice(actions, [0, i], [-1, 1]))
            #toss entries
            actions = []
            for i in range(25):
                if i not in [0, 1, 10, 12, 13, 17, 19, 21, 22]:
                    actions.append(features[i])
            actions = tf.concat(actions, 1)

        elif self.normalize_actions is not None:
            raise TypeError('Unknown normalization type for actions')

        if not self.use_object_ids: 
            #TODO ONLY USE WITH CUSTOM!
            if self.normalize_actions not in ['custom', 'toss_zeros']:
                raise TypeError('use_object_ids = False only allowed for \
                                 normalize_actions = \'custom\'/\'toss_zeros\'')
            actions = tf.concat([
# EGO MOTION
#                  tf.slice(actions, [0,  1], [-1, 6]),

# ONLY ONE ACTION
                  tf.slice(actions, [0,  5], [-1, 6]),
# INCLUDE ACTION ID

# NO OBJ IDS, TELEPORT, FORCE_Z, TORQUE_X, TORQUE_Z
#                 tf.slice(actions, [0,  1], [-1, 8]),
#                 tf.slice(actions, [0, 11], [-1, 1]),
#                 tf.slice(actions, [0, 14], [-1, 2]),

# NO OBJ IDS
#                tf.slice(actions, [0,  0], [-1, 13]),
#                tf.slice(actions, [0, 14], [-1, 8]),
#                tf.slice(actions, [0, 23], [-1, -1]),
            ], 1)
            # now shape is 23 instead 25 since object ids were removed
        return actions

    def set_data_shapes(self, data):
        for i in range(len(data)):
            for k in data[i]:
                # set shape[0] to batch size for all entries
                shape = data[i][k].get_shape().as_list()
                shape[0] = self.batch_size
                data[i][k].set_shape(shape)
        return data

    def init_ops(self):
        self.input_ops = super(FuturePredictionData, self).init_ops()

        # make sure batch size shapes of tensors are set
        self.input_ops = self.set_data_shapes(self.input_ops)

        for i in range(len(self.input_ops)):
            # remove first image pair as it is teleporting
            if(self.is_remove_teleport):
                self.input_ops[i] = self.remove_teleport(self.input_ops[i])
            # convert action position into gaussian channel
            if self.action_matrix_radius is not None:
                if self.normalize_actions not in ['custom', 'toss_zeros'] or \
                                   self.use_object_ids is True:
                    raise TypeError('action_matrix_radius can only be used if  \
                    self.normalize_actions = \'custom\'/\'toss_zeros\' \
                    and self.use_object_ids = False')
                self.input_ops[i] = self.convert_to_action_matrix(self.input_ops[i])

            # create image pairs / sequences
            if (self.output_format['images'] == 'pairs'):
                self.input_ops[i] = self.create_image_pairs(self.input_ops[i])
            
            elif (self.output_format['images'] == 'sequence'):
                self.input_ops[i] = self.create_image_sequence(self.input_ops[i])
            else:
                raise KeyError('Unknown image output format')

            if (self.output_format['actions'] == 'pairs'):
                self.input_ops[i] = self.create_action_pairs(self.input_ops[i])

            elif (self.output_format['actions'] == 'sequence'):
                self.input_ops[i] = self.create_action_sequence(self.input_ops[i])
            else:
                raise KeyError('Unknown action output format')
            
            # remove entries according to filters
            if self.filters is not None:
                self.input_ops[i] = self.apply_filters(self.input_ops[i])

            # add ids
            #self.input_ops[i] = self.add_ids(self.input_ops[i])
        return self.input_ops

    def apply_filters(self, data):
        # delta time which corresponds to the sequence length
        delta_time = tf.constant(self.min_time_difference, dtype=tf.int32)

        # filter actions and images based on provided binary labels
        if self.output_format['actions'] is 'sequence' or \
                    self.output_format['images'] is 'sequence':
            output_format_func = self.create_sequence
        else:
            output_format_func = self.create_pairs

        for f in self.filters:
            # decode filter
            shape = data[f].get_shape().as_list()
            data[f] = tf.decode_raw(data[f], tf.int32)
            data[f] = tf.reshape(data[f], shape)
            # set the correct shape
            if len(shape) < 2:
                # make sure filter has second dimension (important when pair output)
                data[f] = tf.expand_dims(data[f], -1)
                shape = data[f].get_shape().as_list()
            shape[1] = 1
            data[f].set_shape(shape)
            # format filters in the same way as data
            data = output_format_func(data, f, 'future_' + f)
            # check if ALL binary labels within sequence are not zero
            filter_sum = tf.reduce_sum(data[f], 1)
            pos_idx = tf.equal(filter_sum, delta_time)
            # gather positive examples for each data entry
            for k in data:
                shape = data[k].get_shape().as_list()
                shape[0] = -1
                data[k] = tf.gather(data[k], tf.where(pos_idx))
                data[k] = tf.reshape(data[k], shape)
            # remove filter from dict such that it is not enqueued
            data.pop(f)
            data.pop('future_' + f)

        return data

    def create_gaussian_kernel(self, size, center=None, fwhm = 10.0):
        '''
        size: kernel size
        fwhm: full-width-half-maximum (effective radius)
        center: kernel_center
        '''
        batch_size = size[0]
        x = tf.range(0, size[1], 1, dtype=tf.float32)
        x = tf.tile(x, [batch_size])
        x = tf.reshape(x, [batch_size, 1, size[1]]) #column vector
        y = tf.range(0, size[2], 1, dtype=tf.float32)
        y = tf.tile(y, [batch_size])
        y = tf.reshape(y, [batch_size, size[2], 1]) #row vector

        if center is None:
            x0 = tf.constant(size[1] // 2)
            y0 = tf.constant(size[2] // 2)
        else:
            x0 = tf.slice(center, [0, 0], [-1, 1])
            y0 = tf.slice(center, [0, 1], [-1, 1]) 

        x0 = tf.reshape(x0, [batch_size, 1, 1])
        y0 = tf.reshape(y0, [batch_size, 1, 1])
        gauss = tf.exp(-4.0*tf.log(2.0) * ((x-x0)**2.0 + (y-y0)**2.0) / fwhm**2.0)
        y0 = tf.reshape(y0, [batch_size, 1, 1])
        gauss = tf.exp(-4.0*tf.log(2.0) * ((x-x0)**2.0 + (y-y0)**2.0) / fwhm**2.0)
        size.append(1) # append channel dimension
        gauss = tf.reshape(gauss*255, size) # scale to whole uint8 range
        return tf.cast(gauss, tf.uint8)

    def convert_to_action_matrix(self, data):
        action_pos = tf.slice(data[self.actions], [0, 4], [-1, 2])

        #undo normalization
        if self.normalize_actions is 'standard':
            action_mean = tf.slice(self.stats['mean'][self.actions], [14], [2])
            action_std = tf.slice(self.stats['std'][self.actions], [14], [2])
            action_pos = action_pos * (action_std + self.epsilon) + action_mean
            
        elif self.normalize_actions is 'minmax':
            action_min = tf.slice(self.stats['min'][self.actions], [14], [2])
            action_max = tf.slice(self.stats['max'][self.actions], [14], [2])
            action_pos = action_pos * (action_max - action_min) + action_min
            #action_pos = tf.cast(action_pos, tf.int32)
        elif self.normalize_actions is 'custom':
            action_pos = tf.multiply(action_pos, 255)
        
        image_shape = data[self.images].get_shape().as_list()
        image_shape[-1] += 1

        gauss_img = self.create_gaussian_kernel(image_shape[0:3], \
                action_pos, self.action_matrix_radius)
        data[self.images] = tf.concat([data[self.images], gauss_img], 3)

        data[self.actions] = tf.slice(data[self.actions], [0, 0], [-1, 4])

        return data

    def add_ids(self, data):
        batch_size_loaded = data[self.images].get_shape().as_list()[0]
        data['id'] = tf.range(self.example_counter, 
                self.example_counter + batch_size_loaded, 1)

        shape = data['id'].get_shape().as_list()
        shape[0] = data[self.images].get_shape().as_list()[0]
        data['id'].set_shape(shape)

        self.example_counter = self.example_counter + batch_size_loaded
        return data

    def remove_teleport(self, data):
        for k in data:
            size = data[k].get_shape().as_list()
            size[0] -= 1
            begin = [1] + [0] * (len(size) - 1)
            data[k] = tf.slice(data[k], begin, size)

        return data

    def create_pairs(self, data, current_key, future_key):
        size = data[current_key].get_shape().as_list()
        size[0] -= self.min_time_difference
        begin = [self.min_time_difference] + [0] * (len(size) - 1)
        data[future_key] = tf.slice(data[current_key], begin, size)

        begin = [0] * len(size)
        data[current_key] = tf.slice(data[current_key], begin, size)

        return data

    def create_sequence(self, data, current_key, future_key):
        size = data[current_key].get_shape().as_list()
        size[0] -= self.min_time_difference
        begin = [self.min_time_difference] + [0] * (len(size) - 1)
        data[future_key] = tf.slice(data[current_key], begin, size)

        shifts = []
        for i in range(self.min_time_difference):
            begin = [i] + [0] * (len(size) - 1)
            shifts.append(tf.slice(data[current_key], begin, size))

        data[current_key] = tf.concat(shifts, (len(size) - 1))

        return data

    def create_image_pairs(self, data):
        return self.create_pairs(data, self.images, 'future_' + self.images)

    def create_action_pairs(self, data):
        return self.create_pairs(data, self.actions, 'future_' + self.actions)

    def create_image_sequence(self, data):
        return self.create_sequence(data, self.images, 'future_' + self.images)

    def create_action_sequence(self, data):
        return self.create_sequence(data, self.actions, 'future_' + self.actions)
