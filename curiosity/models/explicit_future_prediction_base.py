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
    print('center and magnitude')
    print(center)
    print(magnitude)

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
                 normalization=None,
                 objects_to_include = None, 
                 *args,
                 **kwargs):
        self.inputs = inputs
        self.normalization = normalization

        # store reference to not normed inputs for not normed pixel positions
        inputs_not_normed = inputs

        if self.normalization is not None:
            self.inputs = self.normalization.normalize(self.inputs)

        #requiring that normals is in the data
        image_shape = self.inputs['normals'].get_shape().as_list()
        gaussians = []

        time_before_shape = copy.deepcopy(image_shape)
        time_before_shape[1] = time_seen
        time_after_shape = copy.deepcopy(image_shape)
        time_after_shape[1] = image_shape[1] - time_seen

    	#objects_to_include should really be fed in as a tensor input generated randomly from data provider.
    	if objects_to_include is None:
    		#sets only the acted-on object and the biggest other object for object of interest
	    	centroids = [inputs_not_normed['object_data'][:,:time_seen,obj_num, 8:10] for obj_num in [0,1]]
	    	poses = [inputs_not_normed['object_data'][:, :time_seen, obj_num, 1:5] for obj_num in [0,1]]
        else:
	    	raise NotImplementedError('Need to make fancier slicing for general case')
    	
        for (centroid, pose) in zip(centroids, poses):
    		pose = tf.unstack(pose, axis = 2)
    		for pose_val in pose:
    			gaussians.append(create_gaussian_channel(time_before_shape, centroid, pose_val))

        self.inputs['object_data_seen'] = tf.concat(gaussians, axis = 4)


        gaussians = []

	    # add actions up to the time seen
        centroid = inputs_not_normed['object_data'][:, :time_seen, 0, 8:10]
        force = inputs_not_normed['actions'][:, :time_seen, :6]
        force = tf.unstack(force, axis = 2)
        for f in force:
            gaussians.append(create_gaussian_channel(time_before_shape, centroid, f))

        self.inputs['actions_seen'] = tf.concat(gaussians, axis = 4)

        gaussians = []
	    #add actions after time seen, but in last known position of acted-on object (otherwise this would give it away...)
        last_known_centroid = inputs_not_normed['object_data'][:, time_seen - 1 : time_seen, 0, 8:10]
        centroid = tf.concat([last_known_centroid for _ in range(time_seen, image_shape[1])], axis = 1)
        force = inputs_not_normed['actions'][:, time_seen:, :6]
        force = tf.unstack(force, axis = 2)
        for f in force:
            gaussians.append(create_gaussian_channel(time_after_shape, centroid, f))

        self.inputs['actions_future'] = tf.concat(gaussians, axis = 4)
        fut_pose = self.inputs['object_data'][:, time_seen : , 0:2, 1:5]
        fut_pos = self.inputs['object_data'][:, time_seen : , 0:2, 8:10]
        fut_dat = tf.concat([fut_pose, fut_pos], axis = 3)

        self.inputs['object_data_future'] = fut_dat

        seen_pose = self.inputs['object_data'][:, : time_seen, 0:2, 1:5]
        seen_pos = self.inputs['object_data'][:, : time_seen, 0:2, 8:10]
        seen_dat = tf.concat([fut_pose, fut_pos], axis = 3)

        self.inputs['object_data_seen_1d'] = seen_dat









