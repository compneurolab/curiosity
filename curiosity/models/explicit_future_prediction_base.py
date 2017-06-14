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
	elif 'depths' in inputs_not_normed:
	    image_shape = inputs_not_normed['depths'].get_shape().as_list()
        else:
	    print('CMON: ' + str(inputs_not_normed.keys()))
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
            time_seen = None, scale_down_height = None, scale_down_width = None, add_depth_gaussian = False,
		store_jerk = False, hack_jerk_norm = True, depth_cutoff = None,
		get_actions_map = False,
		get_segmentation = False,
                get_hacky_segmentation_map = False,
                *args,  **kwargs):
        self.inputs = {}
        self.normalization_method = dict(normalization_method)
        assert time_seen is not None

        # store reference to not normed inputs for not normed pixel positions
        inputs_not_normed = dict(inputs)


        if 'object_data' in self.normalization_method and self.normalization_method['object_data'] == 'screen_normalize':
            screen_normalize = True
            self.normalization_method.pop('object_data')
            self.normalization_method['object_data'] = 'standard'
        else:
            screen_normalize = False

        if self.normalization_method is not None:
            if stats_file is not None:
            	normalization = tb.Normalizer(stats_file, self.normalization_method)
            	normed_inputs = normalization.normalize(inputs)
            else:
		raise Exception('Not implememted!')

        obj_data_shape = inputs_not_normed['object_data'].get_shape().as_list()
        batch_size = obj_data_shape[0]
        long_len = obj_data_shape[1]

        if 'normals' in inputs_not_normed:
            im_sh = inputs_not_normed['normals'].get_shape().as_list()
            img_height = im_sh[2]
            img_width = im_sh[3]
	elif 'depths' in inputs_not_normed:
            im_sh = inputs_not_normed['depths'].get_shape().as_list()
            img_height = im_sh[2]
            img_width = im_sh[3]
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
            if add_depth_gaussian:
                print('Adding depth gaussian!')
                depths = [tf.tanh(normed_inputs['object_data'][:, : time_seen, obj_num, 7:8]) for obj_num in objects_to_include]
                poses = [tf.concat([pose, depth], axis = -1) for (pose, depth) in zip(poses, depths)]

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
        fut_pos = inputs_not_normed['object_data'][:, time_seen : , 0:1, 8:10]

        screen_dims = [img_height, img_width]
        if screen_normalize:
            fut_pos = tf.concat([2. * (fut_pos[:, :, :, i:i+1] - float(screen_dims[i]) / 2.) / float(screen_dims[i]) for i in [0,1]], axis = 3)
            fut_pos = tf.tanh(fut_pos)

        fut_dat = tf.concat([fut_pose, fut_pos], axis = 3)

        self.inputs['object_data_future'] = fut_dat

        seen_pose = inputs_not_normed['object_data'][:, : time_seen, 0:1, 1:5] 
        seen_pos = inputs_not_normed['object_data'][:, : time_seen, 0:1, 8:10]
        if screen_normalize:
            seen_pos = tf.concat([2. * (seen_pos[:, :, :, i:i+1] - float(screen_dims[i]) / 2.) / float(screen_dims[i]) for i in [0,1]], axis = 3)
            seen_pos = tf.tanh(seen_pos)

        seen_dat = tf.concat([seen_pose, seen_pos], axis = 3) #BATCH_SIZE x time_seen x 2 x 6

        self.inputs['object_data_seen_1d'] = seen_dat

        acted_on_ids = tf.cast(inputs_not_normed['object_data'][:, :time_seen, 0:1, 0:1], tf.int32)
        acted_on_ids = tf.expand_dims(acted_on_ids, 3)
        acted_on_ids = tf.tile(acted_on_ids, [1, 1, img_height, img_width, 1])


        self.inputs['depth_seen'] = tf.tanh(normed_inputs['object_data'][:, : time_seen, 0:1, 7:8])

        for desc in ['objects', 'objects2']:
            if desc in inputs_not_normed:
                objects = tf.cast(inputs_not_normed[desc], tf.int32)
                objects = objects[:, :, :, :, 0:1] * (256**2) + objects[:, :, :, :, 1:2] * 256 + objects[:, :, :, :, 2:3]
                objects = objects[:, :time_seen]
                self.inputs[desc] = tf.cast(tf.equal(acted_on_ids, objects), tf.float32)

        for desc in ['depths', 'depths2']:
            if desc in inputs_not_normed:
                self.inputs[desc + '_raw'] =  inputs_not_normed[desc]
                depths = tf.cast(inputs_not_normed[desc], tf.float32)
                depths = (depths[:,:,:,:,0:1] * 256 + depths[:,:,:,:,1:2] + \
                        depths[:,:,:,:,2:3] / 256.0) / 1000.0 
                depths /= 17.32 # normalization
                depths = depths[:, :time_seen]
                self.inputs[desc] = depths

        #forces us to not manually normalize, should be reasonable, makes for easier viz.
        for desc in ['normals', 'normals2', 'images', 'images2']:
            if desc in inputs_not_normed:
                self.inputs[desc] = tf.cast(inputs_not_normed[desc], tf.float32) / 255.

        for desc in ['vels', 'vels2', 'jerks', 'jerks2', 'accs', 'accs2',
               'vels_curr', 'vels_curr2', 'jerks_curr', 'jerks_curr2',
                'accs_curr', 'accs_curr2']:
            if desc in inputs_not_normed:
                self.inputs[desc] = tf.cast(inputs_not_normed[desc], tf.int32)
                self.inputs[desc + '_normed'] = tf.cast(inputs_not_normed[desc], 
                        tf.float32) / 255.


        self.inputs['reference_ids'] = inputs_not_normed['reference_ids']
        #TODO: in case of a different object being acted on, should maybe have action position stuff in for seen times
        if 'actions' in inputs:
		self.inputs['actions_no_pos'] = normed_inputs['actions'][:, :, :, :6]

        self.inputs['master_filter'] = inputs_not_normed['master_filter']

        # create segmented action maps
	if get_actions_map or get_segmentation or get_hacky_segmentation_map:
        	objects = tf.cast(inputs_not_normed['objects'], tf.int32)
        	shape = objects.get_shape().as_list()
        	objects = tf.unstack(objects, axis=len(shape)-1)
        	objects = objects[0] * (256**2) + objects[1] * 256 + objects[2]
	if get_actions_map:
        	forces = self.inputs['actions_no_pos']
		actions_map_list = []
        	for i in range(2):
			action_id = tf.expand_dims(inputs_not_normed['actions'][:,:,i,8], axis=2)
        		action_id = tf.cast(tf.reshape(tf.tile(action_id, 
            				[1, 1, shape[2] * shape[3]]), shape[:-1]), tf.int32)
        		actions = tf.cast(tf.equal(objects, action_id), tf.float32)
        		actions = tf.tile(tf.expand_dims(actions, axis=4), [1,1,1,1,6])
       			actions *= tf.expand_dims(tf.expand_dims(forces[:, :, i, :], 2), 2)
			actions_map_list.append(tf.expand_dims(actions, -1))
        	self.inputs['actions_map'] = tf.concat(actions_map_list, -1)
	if get_segmentation:
		segmentation_list = []
		action_ids = inputs_not_normed['actions'][:, :, :, 8]
		for i in range(2):
			action_id_pic = tf.expand_dims(action_ids[:, :, i], axis = 2)
			action_id_pic = tf.cast(tf.reshape(tf.tile(action_id_pic, [1, 1, shape[2] * shape[3]]), shape[:-1]), tf.int32)
			segmentation = tf.equal(objects, action_id_pic)
			segmentation_list.append(tf.expand_dims(segmentation, -1))
		self.inputs['segmentation'] = tf.cast(tf.concat(segmentation_list, -1), tf.int32)
		self.inputs['action_ids'] = action_ids
        if get_hacky_segmentation_map:
            # couch == id 23 and microwave == id 24
            segmentation_list = [
                    tf.expand_dims(tf.cast(tf.equal(objects, 23), tf.float32) * 0.5, -1),
                    tf.expand_dims(tf.cast(tf.equal(objects, 24), tf.float32) * 1, -1)]
            self.inputs['segmentation_map'] = tf.concat(segmentation_list, -1)


	if store_jerk:
		#jerk
                pos_all = inputs_not_normed['object_data'][:, :, :, 5:8]
		vel_all = pos_all[:, 1:] - pos_all[:, :-1]
		acc_all = vel_all[:, 1:] - vel_all[:, :-1]
		jerk_all = acc_all[:, 1:] - acc_all[:, :-1]
		assert(jerk_all.get_shape().as_list()[1] == 1)
		jerk_all = jerk_all[:,0]
		if hack_jerk_norm:
			print('doing hackish jerk norm!')
			#jerk_all = tf.maximum(tf.minimum(jerk_all, 3.), -3.)/3.
                        #jerk_all = tf.clip_by_value(jerk_all, -0.6, 0.6) / 0.13130946
                        jerk_all = tf.clip_by_value(jerk_all, -0.6, 0.6) / 0.6
                self.inputs['jerk'] = jerk_all[:, 0]
                self.inputs['jerk_all'] = jerk_all

                pos_ids = tf.cast(inputs_not_normed['object_data'][:,:,:,0], tf.int32) 
                pos_ids = tf.unstack(pos_ids, axis=2)
                jerk_all = tf.unstack(jerk_all, axis=1)
                # velocity at t=0 equals position
                vel_all = tf.concat([pos_all[:, 0:1], vel_all], axis=1)
                vel_all = tf.unstack(vel_all, axis=2)

                for objects_key in ['objects', 'objects2']:
                    if objects_key not in inputs_not_normed:
                        continue
                    objects = tf.cast(inputs_not_normed[objects_key], tf.int32)
                    shape = objects.get_shape().as_list()
                    objects = tf.unstack(objects, axis=len(shape)-1)
                    objects = objects[0] * (256**2) + objects[1] * 256 + objects[2]
                    # calculate jerk map
                    jerk_map = tf.zeros([shape[0], shape[2], shape[3], 3], tf.float32)
                    # calculate velocity maps
                    vel_maps = tf.zeros([shape[0], shape[1]-1, shape[2], shape[3], 3], 
                            tf.float32)
                    for pos_id, jerk, vel in zip(pos_ids, jerk_all, vel_all):
                        pos_id = tf.expand_dims(tf.expand_dims(pos_id, 2), 2)
                        segment = tf.tile(tf.expand_dims(tf.cast(tf.equal(
                            pos_id, objects), tf.float32), 4), [1,1,1,1,3])
                        jerk_map += segment[:,-2] * \
                                tf.expand_dims(tf.expand_dims(jerk, 1), 1)
                        vel_maps += segment[:,:-1] * \
                                tf.expand_dims(tf.expand_dims(vel[:,:-1], 2), 2)
                    if objects_key is 'objects':
                        self.inputs['jerk_map'] = jerk_map
                        self.inputs['vel_map'] = vel_maps
                    elif objects_key is 'objects2':
                        self.inputs['jerk_map2'] = jerk_map
                        self.inputs['vel_map2'] = vel_maps
