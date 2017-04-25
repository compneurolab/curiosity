'''
Future prediction base. Like threeworld_base, but we
(1) want to draw only a subset of the objects, and in different channels,
(2) want to draw later actions in the last known position of the object being acted on (with the centroid of the object projected onto the screen, rather than the actual action position), and
(3) want to give nice preprocessed, flattened output.
'''

import threeworld_base as tb
import copy
import tensorflow as tf


def create_gaussian_channel(size, 
                            center=None, 
                            magnitude=None,
                            dtype = tf.float32,
                            fwhm = 10.0):
    '''
    size: kernel size
    fwhm: full-width-half-maximum (effective radius)
    center: kernel_center
    '''
    batch_size = size[0]
    sequence_len = size[1]
    width = size[2]
    height = size[3]
    channels = size[4]
    x = tf.range(0, width, 1, dtype=tf.float32)
    x = tf.tile(x, [batch_size * sequence_len])
    x = tf.reshape(x, [batch_size, sequence_len, width, 1, 1]) #column vector
    y = tf.range(0, height, 1, dtype=tf.float32)
    y = tf.tile(y, [batch_size * sequence_len])
    y = tf.reshape(y, [batch_size, sequence_len, 1, height, 1]) #row vector

    x0s = []
    y0s = []
    if center is None:
        x0s.append(tf.constant(width // 2))
        y0s.append(tf.constant(height // 2))
    elif isinstance(center, list):
        for c in center:
            x0s.append(tf.slice(c, [0, 0, 0], [-1, -1, 1]))
            y0s.append(tf.slice(c, [0, 0, 1], [-1, -1, 1]))
    else:
        x0s.append(tf.slice(center, [0, 0, 0], [-1, -1, 1]))
        y0s.append(tf.slice(center, [0, 0, 1], [-1, -1, 1]))

    mags = []
    if magnitude is None:
        mags.append(tf.ones([1, 1, 1]))
    elif isinstance(magnitude, list):
        mags = magnitude
    else:
        mags.append(magnitude)

    # create one dimensional channel with gaussians at [x0s, y0s]
    gauss = tf.zeros(size[0:4] + [1])
    for x0, y0, mag in zip(x0s, y0s, mags):
        x0 = tf.reshape(x0, [batch_size, sequence_len, 1, 1, 1])
        y0 = tf.reshape(y0, [batch_size, sequence_len, 1, 1, 1])
        mag = tf.reshape(mag, [batch_size, sequence_len, 1, 1, 1])
        gauss += tf.exp(-4.0*tf.log(2.0) * \
                ((x-x0)**2.0 + (y-y0)**2.0) / fwhm**2.0) * mag
    return tf.cast(gauss, dtype)



class FuturePredictionBaseModel:
    def __init__(self,
                 inputs,
                 time_seen,
                 normalization_method = None,
                 stats_file = None,
                 objects_to_include = None,
                 add_gaussians = True,
                 img_height = None,
                 img_width = None,
                 *args,
                 **kwargs):
        self.inputs = {}
        self.normalization_method = dict(normalization_method)

        # store reference to not normed inputs for not normed pixel positions
        inputs_not_normed = dict(inputs)


        if 'object_data' in self.normalization_method and self.normalization_method['object_data'] == 'screen_normalize':
            screen_normalize = True
            self.normalization_method.pop('object_data')
        else:
            screen_normalize = False

        if self.normalization_method is not None:
            assert stats_file is not None
            normalization = tb.Normalizer(stats_file, self.normalization_method)
            normed_inputs = normalization.normalize(inputs)

        if 'normals' in inputs_not_normed:
            image_shape = inputs_not_normed['normals'].get_shape().as_list()
        else:
            assert img_height is not None and img_width is not None
            #requires object_data in there...
            obj_data_shape = inputs_not_normed['object_data'].get_shape().as_list()
            batch_size = obj_data_shape[0]
            seq_len = obj_data_shape[1]
            image_shape = [batch_size, seq_len, img_height, img_width, 3]
        if add_gaussians:
            gaussians = []

            #er could shallow copy here...
            time_before_shape = copy.deepcopy(image_shape)
            time_before_shape[1] = time_seen
            time_after_shape = copy.deepcopy(image_shape)
            time_after_shape[1] = image_shape[1] - time_seen

        	#objects_to_include should really be fed in as a tensor input generated randomly from data provider.
            if objects_to_include is None:
                objects_to_include = [0]
            #sets only the acted-on object and the biggest other object for object of interest
            else:
            	raise NotImplementedError('Need to make fancier slicing for general case')
        	
            centroids = [inputs_not_normed['object_data'][:,:time_seen,obj_num, 8:10] for obj_num in objects_to_include]
            poses = [inputs_not_normed['object_data'][:, :time_seen, obj_num, 1:5] for obj_num in objects_to_include]


            for (centroid, pose) in zip(centroids, poses):
        		pose = tf.unstack(pose, axis = 2)
        		for pose_val in pose:
        			gaussians.append(create_gaussian_channel(time_before_shape, centroid, pose_val))

            self.inputs['object_data_seen'] = tf.concat(gaussians, axis = 4)


            gaussians = []

    	    # add actions up to the time seen
            centroid = inputs_not_normed['object_data'][:, :time_seen, 0, 8:10]
            #normalize!
            force = normed_inputs['actions'][:, :time_seen, :6]
            force = tf.unstack(force, axis = 2)
            for f in force:
                gaussians.append(create_gaussian_channel(time_before_shape, centroid, f))

            self.inputs['actions_seen'] = tf.concat(gaussians, axis = 4)

            gaussians = []
    	    #add actions after time seen, but in last known position of acted-on object (otherwise this would give it away...)
            last_known_centroid = inputs_not_normed['object_data'][:, time_seen - 1 : time_seen, 0, 8:10]
            centroid = tf.concat([last_known_centroid for _ in range(time_seen, image_shape[1])], axis = 1)
            force = normed_inputs['actions'][:, time_seen:, :6]
            force = tf.unstack(force, axis = 2)
            for f in force:
                gaussians.append(create_gaussian_channel(time_after_shape, centroid, f))

            self.inputs['actions_future'] = tf.concat(gaussians, axis = 4)
        #normalize?
        fut_pose = inputs_not_normed['object_data'][:, time_seen : , 0:1, 1:5]
        #normalize! use std method
        fut_pos = normed_inputs['object_data'][:, time_seen : , 0:1, 8:10]
        if screen_normalize:
            fut_pos = tf.concat([2. * (fut_pos[:, :, :, i:i+1] - float(image_shape[i+2]) / 2.) / float(image_shape[i+2]) for i in [0,1]], axis = 3)
            fut_pos = tf.tanh(fut_pos)

        fut_dat = tf.concat([fut_pose, fut_pos], axis = 3)

        self.inputs['object_data_future'] = fut_dat

        seen_pose = inputs_not_normed['object_data'][:, : time_seen, 0:1, 1:5] 
        seen_pos = normed_inputs['object_data'][:, : time_seen, 0:1, 8:10]
        if screen_normalize:
            seen_pos = tf.concat([2. * (seen_pos[:, :, :, i:i+1] - float(image_shape[i+2]) / 2.) / float(image_shape[i+2]) for i in [0,1]], axis = 3)
            seen_pos = tf.tanh(seen_pos)

        seen_dat = tf.concat([seen_pose, seen_pos], axis = 3) #BATCH_SIZE x time_seen x 2 x 6

        self.inputs['object_data_seen_1d'] = seen_dat

        #forces us to not manually normalize, should be reasonable, makes for easier viz.
        if 'normals' in inputs_not_normed:
            self.inputs['normals'] = tf.cast(inputs_not_normed['normals'], tf.float32) / 255.
            self.inputs['normals2'] = tf.cast(inputs_not_normed['normals2'], tf.float32) / 255.

        self.inputs['reference_ids'] = inputs_not_normed['reference_ids']

        self.inputs['actions_no_pos'] = normed_inputs['actions'][:, :, :6]


class ShortLongFuturePredictionBase:

    def __init__(self, inputs, normalization_method = None, stats_file = None,
            objects_to_include = None, add_gaussians = True, img_height = None, img_width = None,
            time_seen = None, scale_down_height = None, scale_down_width = None,
                *args,  **kwargs):
        self.inputs = {}
        self.normalization_method = dict(normalization_method)
        assert time_seen is not None

        # store reference to not normed inputs for not normed pixel positions
        inputs_not_normed = dict(inputs)


        if 'object_data' in self.normalization_method and self.normalization_method['object_data'] == 'screen_normalize':
            screen_normalize = True
            self.normalization_method.pop('object_data')
        else:
            screen_normalize = False

        if self.normalization_method is not None:
            assert stats_file is not None
            normalization = tb.Normalizer(stats_file, self.normalization_method)
            normed_inputs = normalization.normalize(inputs)
        
        obj_data_shape = inputs_not_normed['object_data'].get_shape().as_list()
        batch_size = obj_data_shape[0]
        long_len = obj_data_shape[1]

        if 'normals' in inputs_not_normed:
            im_sh = inputs_not_normed['normals'].get_shape().as_list()
            img_height = im_sh[2]
            img_width = im_sh[3]
            assert time_seen == im_sh[1]
        else:
            assert img_height is not None and img_width is not None

        if add_gaussians:
            gaussians = []

            #er could shallow copy here...
            gaussian_shape = [batch_size, time_seen, img_height, img_width, 1]

            if scale_down_height is not None:
                assert scale_down_width is not None
                gaussian_shape[2] = scale_down_height
                gaussian_shape[3] = scale_down_width
                scale_down_gaussians = float(scale_down_height) / float(img_height)
            else:
                scale_down_gaussians = 1.
            #objects_to_include should really be fed in as a tensor input generated randomly from data provider.
            if objects_to_include is None:
                objects_to_include = [0]
            else:
                raise NotImplementedError('Need to make fancier slicing for general case')
            
            centroids = [scale_down_gaussians * inputs_not_normed['object_data'][:,:time_seen,obj_num, 8:10] for obj_num in objects_to_include]
            poses = [inputs_not_normed['object_data'][:, :time_seen, obj_num, 1:5] for obj_num in objects_to_include]


            for (centroid, pose) in zip(centroids, poses):
                pose = tf.unstack(pose, axis = 2)
                for pose_val in pose:
                    gaussians.append(create_gaussian_channel(gaussian_shape, centroid, pose_val))

            self.inputs['object_data_seen'] = tf.concat(gaussians, axis = 4)


            gaussians = []

            # add actions up to the time seen
            centroid = scale_down_gaussians * inputs_not_normed['object_data'][:, :time_seen, 0, 8:10]
            #normalize!
            force = normed_inputs['actions'][:, :time_seen, :6]
            force = tf.unstack(force, axis = 2)
            for f in force:
                gaussians.append(create_gaussian_channel(gaussian_shape, centroid, f))

            self.inputs['actions_seen'] = tf.concat(gaussians, axis = 4)

            # gaussians = []
            # #add actions after time seen, but in last known position of acted-on object (otherwise this would give it away...)
            # last_known_centroid = inputs_not_normed['object_data'][:, time_seen - 1 : time_seen, 0, 8:10]
            # centroid = tf.concat([last_known_centroid for _ in range(time_seen, image_shape[1])], axis = 1)
            # force = normed_inputs['actions'][:, time_seen:, :6]
            # force = tf.unstack(force, axis = 2)
            # for f in force:
            #     gaussians.append(create_gaussian_channel(time_after_shape, centroid, f))

        fut_pose = inputs_not_normed['object_data'][:, time_seen : , 0:1, 1:5]
        #normalize! use std method
        fut_pos = normed_inputs['object_data'][:, time_seen : , 0:1, 8:10]
        screen_dims = [img_height, img_width]
        if screen_normalize:
            fut_pos = tf.concat([2. * (fut_pos[:, :, :, i:i+1] - float(screen_dims[i]) / 2.) / float(screen_dims[i]) for i in [0,1]], axis = 3)
            fut_pos = tf.tanh(fut_pos)

        fut_dat = tf.concat([fut_pose, fut_pos], axis = 3)
        print('future dat')
        print(fut_dat)

        self.inputs['object_data_future'] = fut_dat

        seen_pose = inputs_not_normed['object_data'][:, : time_seen, 0:1, 1:5] 
        seen_pos = normed_inputs['object_data'][:, : time_seen, 0:1, 8:10]
        if screen_normalize:
            seen_pos = tf.concat([2. * (seen_pos[:, :, :, i:i+1] - float(screen_dims[i]) / 2.) / float(screen_dims[i]) for i in [0,1]], axis = 3)
            seen_pos = tf.tanh(seen_pos)

        seen_dat = tf.concat([seen_pose, seen_pos], axis = 3) #BATCH_SIZE x time_seen x 2 x 6

        self.inputs['object_data_seen_1d'] = seen_dat

        #forces us to not manually normalize, should be reasonable, makes for easier viz.
        if 'normals' in inputs_not_normed:
            self.inputs['normals'] = tf.cast(inputs_not_normed['normals'], tf.float32) / 255.
            self.inputs['normals2'] = tf.cast(inputs_not_normed['normals2'], tf.float32) / 255.

        self.inputs['reference_ids'] = inputs_not_normed['reference_ids']
        #TODO: in case of a different object being acted on, should maybe have action position stuff in for seen times
        self.inputs['actions_no_pos'] = normed_inputs['actions'][:, :, :6]

        self.inputs['master_filter'] = inputs_not_normed['master_filter']


