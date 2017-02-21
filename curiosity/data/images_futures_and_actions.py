from tfutils.data import TFRecordsParallelByFileProvider
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import copy
import os

class FuturePredictionData(TFRecordsParallelByFileProvider):
    example_counter = tf.constant(0)

    epsilon = np.array(1e-1).astype(np.float32)

    actions_mean = np.array(
      [  3.90625000e-03,   0.00000000e+00,  -2.44093345e-04, #teleport, vel_x, vel_y
         5.20533161e-02,   1.22121812e+02,  -3.34609385e-05, #vel_z, ang_x, ang_y
        -7.55522251e+02,  -1.47485679e-01,   1.06664636e+02, #ang_z, a1_fx, a1_fy
         1.28125378e-02,   0.00000000e+00,  -2.27804319e-02, #a1_fz, a1_tx, a1_ty,
         0.00000000e+00,   3.44634385e+01,   7.06908594e+01, #a1_tz, a1_id, a1_pos_x,
         6.58260645e+01,   3.78274760e-02,   0.00000000e+00, #a1_pos_y, a2_fx, a2_fy,
        -1.28125378e-02,   0.00000000e+00,  -1.32991500e-03, #a2_fz, a2_tx, a2_ty,
         0.00000000e+00,   6.64200195e-01,   1.27456543e+00, #a2_tz, a2_id, a2_pos_x,
         1.30035547e+00]).astype(np.float32)                 #a2_pos_y

    actions_std = np.array(
      [  6.23778102e-02,   0.00000000e+00,   4.53425576e-03,  
         1.01547240e-01,   2.22054444e+06,   6.04687621e-02,
         1.43378085e+06,   1.27678463e+02,   3.23207041e+03,
         1.95972036e+01,   0.00000000e+00,   1.37277482e+01,
         0.00000000e+00,   6.96205264e+01,   1.26656184e+02,
         1.27864069e+02,   2.00925928e+01,   0.00000000e+00,
         1.95972036e+01,   0.00000000e+00,   6.21731960e-01,
         0.00000000e+00,   1.07432982e+01,   1.84946833e+01,
         2.16857321e+01]).astype(np.float32)

    def __init__(self,
                 data_path,
                 batch_size=256,
                 crop_size=None,
                 min_time_difference=1, # including, also specifies fixed time
                 output_format={'images': 'pairs', 'actions': 'sequence'},
                 use_object_ids=True,
                 normalize_actions=True,
                 action_matrix_radius=None,
                 is_remove_teleport=True,
                 n_threads=4,
                 *args,
                 **kwargs):

        self.orig_shape = None
        self.output_format = output_format
        self.use_object_ids = use_object_ids
        self.normalize_actions = normalize_actions
        self.is_remove_teleport = is_remove_teleport
        self.action_matrix_radius = action_matrix_radius

        if int(min_time_difference) < 1:
   	    self.min_time_difference = 1
	    print("The minimum time difference has to be at least 1, " \
	        + "and thus was set to 1.") 
        else:
	    self.min_time_difference = int(min_time_difference)

        self.batch_size = batch_size
        if self.batch_size <= self.min_time_difference:
            raise IndexError('batch_size has to be bigger than min_time_difference!')

        self.images = 'images'
        self.actions = 'parsed_actions' #'actions'

        self.source_paths = [os.path.join(data_path, self.images),
                             os.path.join(data_path, self.actions)]

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
        return images

    def postproc_parsed_actions(self, actions):
        actions = tf.decode_raw(actions, tf.float64)
        actions = tf.cast(actions, tf.float32)

        act_shape = actions.get_shape().as_list()
        act_shape[1] = 25
        actions.set_shape(act_shape)

        #normalize actions
        if self.normalize_actions:
            actions = (actions - self.actions_mean) / (self.actions_std + self.epsilon)

        if not self.use_object_ids:
            # object ids are at columns 13 and 22, thus remove those columns
            actions = tf.concat(1, [
# EGO MOTION
#                  tf.slice(actions, [0,  1], [-1, 6]),

# ONLY ONE ACTION
                  tf.slice(actions, [0,  7], [-1, 6]),
# INCLUDE ACTION ID
                  tf.slice(actions, [0, 14], [-1, 2]),

# NO OBJ IDS, TELEPORT, FORCE_Z, TORQUE_X, TORQUE_Z
#                 tf.slice(actions, [0,  1], [-1, 8]),
#                 tf.slice(actions, [0, 11], [-1, 1]),
#                 tf.slice(actions, [0, 14], [-1, 2]),

# NO OBJ IDS
#                tf.slice(actions, [0,  0], [-1, 13]),
#                tf.slice(actions, [0, 14], [-1, 8]),
#                tf.slice(actions, [0, 23], [-1, -1]),
            ])
            # now shape is 23 instead 25 since object ids were removed
        return actions

    def set_data_shapes(self, data):
        for i in range(len(data)):
            # set image batch size
            img_shape = data[i][self.images].get_shape().as_list()
            img_shape[0] = self.batch_size
            data[i][self.images].set_shape(img_shape)
            # set action batch size
            act_shape = data[i][self.actions].get_shape().as_list()
            act_shape[0] = self.batch_size
            data[i][self.actions].set_shape(act_shape)
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

            # add ids
            #self.input_ops[i] = self.add_ids(self.input_ops[i])
        return self.input_ops

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
        action_pos = tf.slice(data[self.actions], [0, 6], [-1, 2])

        #undo normalization
        if self.normalize_actions:
            action_mean = tf.slice(self.actions_mean, [14], [2])
            action_std = tf.slice(self.actions_std, [14], [2])

            action_pos = action_pos * (action_std + self.epsilon) + action_mean
            #action_pos = tf.cast(action_pos, tf.int32)
        
        image_shape = data[self.images].get_shape().as_list()
        image_shape[-1] += 1

        gauss_img = self.create_gaussian_kernel(image_shape[0:3], \
                action_pos, self.action_matrix_radius)
        data[self.images] = tf.concat(3, [data[self.images], gauss_img])

        data[self.actions] = tf.slice(data[self.actions], [0, 0], [-1, 6])

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
        size = data[self.images].get_shape().as_list()
        size[0] -= 1
        begin = [1, 0, 0, 0]
        data[self.images] = tf.slice(data[self.images], begin, size)

        size = data[self.actions].get_shape().as_list()
        size[0] -= 1
        begin = [1, 0]
        data[self.actions] = tf.slice(data[self.actions], begin, size)

        return data

    def create_image_pairs(self, data):
        size = data[self.images].get_shape().as_list()
        size[0] -= self.min_time_difference
        begin = [self.min_time_difference, 0, 0, 0]
        data['future_images'] = tf.slice(data[self.images], begin, size)

        begin = [0,0,0,0]
        data[self.images] = tf.slice(data[self.images], begin, size)

        return data

    def create_action_pairs(self, data):
        size = data[self.actions].get_shape().as_list()
        size[0] -= self.min_time_difference
        begin = [self.min_time_difference, 0]
        data['future_actions'] = tf.slice(data[self.actions], begin, size)

        begin = [0,0]
        data[self.actions] = tf.slice(data[self.actions], begin, size)

        return data
       
    def create_image_sequence(self, data):
        size = data[self.images].get_shape().as_list()
        size[0] -= self.min_time_difference
        begin = [self.min_time_difference, 0, 0, 0]
        data['future_images'] = tf.slice(data[self.images], begin, size)

        shifts = []
        for i in range(self.min_time_difference):
            begin = [i,0,0,0]
            shifts.append(tf.slice(data[self.images], begin, size))

        data[self.images] = tf.concat(3, shifts)

        return data

    def create_action_sequence(self, data):
        size = data[self.actions].get_shape().as_list()
        size[0] -= self.min_time_difference
        begin = [self.min_time_difference, 0]
        data['future_actions'] = tf.slice(data[self.actions], begin, size)        

        shifts = []
        for i in range(self.min_time_difference):
            begin = [i,0]
            shifts.append(tf.slice(data[self.actions], begin, size))

        data[self.actions] = tf.concat(1, shifts)
       
        return data
