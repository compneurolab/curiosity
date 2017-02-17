from tfutils.data import TFRecordsParallelByFileProvider
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import copy
import os

class FuturePredictionData(TFRecordsParallelByFileProvider):
    example_counter = tf.constant(0)
    def __init__(self,
                 data_path,
                 batch_size=256,
                 crop_size=None,
                 min_time_difference=1, # including, also specifies fixed time
                 output_format={'images': 'pairs', 'actions': 'sequence'},
                 use_object_ids=True,
                 action_matrix_radius=None,
                 is_remove_teleport=True,
                 n_threads=4,
                 *args,
                 **kwargs):

        self.orig_shape = None
        self.output_format = output_format
        self.use_object_ids = use_object_ids
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
            shape = [8] #23
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

class FuturePredictionData_DEPRECATED(TFRecordsParallelByFileProvider):
    batch_num = 0
    def __init__(self,
		 data_path,
		 batch_size=256,
		 crop_size=None,
		 min_time_difference=1, # including, also specifies fixed time
		 max_time_difference=2, # excluding
		 random_time=False,
		 *args,
		 **kwargs):
	"""
	A specific reader for ThreeDWorld data stored as a HDF5 file.
	The data will be parsed into an image and action at time t as input 
	and an image and action at time t+1 as output.

	Note: 	
	    This data provider should be used with RandomShuffleQueue since the 
	    data in the HDF5 file is not expected to be shuffled ahead of time.

	Args:
	    - data_path
	        path to ThreeDWorld data

        Kwargs: 
	    - batch_size (int, default: 256)
	        Number of images to return when `next` is called. By default set
	        to 256 since it is expected to be used to load image sequences
   	    - crop_size ([int, int] or None, default: None)
	        For bicubic resizing crop_size = [height, width]. If None, no cropping will occur.
	    - *args, **kwargs
	        Extra arguments for HDF5DataProvider
        """	    
    
        self.images = 'images'
        self.actions = 'parsed_actions' #'actions'
        super(FuturePredictionData, self).__init__(
	    data_path,
	    [self.images, self.actions],
	    batch_size=batch_size,
	    postprocess={self.images: self.postproc_img, self.actions: self.postproc_parsed_actions}, #self.postproc_actions},
	    pad=False,
	    decodelist=[self.images],
	    *args, **kwargs)

	self.crop_size = crop_size

        self.random_time = random_time

   	self.batch_size = batch_size

	self.orig_shape = None
 
        if int(min_time_difference) < 1:
   	    self.min_time_difference = 1
	    print("The minimum time difference has to be at least 1, " \
	        + "and thus was set to 1.") 
        else:
	    self.min_time_difference = int(min_time_difference)

        if int(max_time_difference) < self.min_time_difference:
	    self.max_time_difference = self.min_time_difference + 1
	    print("The maximum time difference has to be bigger than, " \
	        + "the minimum time difference and thus was set to %d." \
	        % self.max_time_difference)
        else:
	    self.max_time_difference = int(max_time_difference)

        self.random_time = random_time

    def postproc_img(self, ims, f):
	# bicubic warping to resize image
	if self.crop_size is not None:
	    images_batch = np.zeros((ims.shape[0], self.crop_size[0], \
					self.crop_size[1], ims.shape[3]))
	    for i in range(len(ims)):
		if i == 0:
		    self.orig_shape = ims[i].shape
		images_batch[i] = np.array( \
		    Image.fromarray(ims[i]).resize( \
			(self.crop_size[0], self.crop_size[1]), Image.BICUBIC))
	    return images_batch
	else:
	    return ims

    def postproc_parsed_actions(self, actions, f):
	parsed_actions = []
	for action in actions:
	    parsed_actions.append(np.fromstring(action, dtype=np.float64))
	return np.array(parsed_actions)

    def postproc_actions(self, actions, f):
	# parse actions into vector 
	parsed_actions = []
	for action in actions:
	    action = json.loads(action)
	    # parsed action vector
	    pact = []
	    # pact[0] : teleport random
	    if 'teleport_random' in action and action['teleport_random'] is True:
		pact.append(1)
	    else:
		pact.append(0)
	    # pact[1:4] : agent velocity
	    if 'vel' in action:
		pact.extend(action['vel'])
	    else:
		pact.extend(np.zeros(3))
	    # pact[4:7] : agent angular velocity
	    if 'ang_vel' in action:
                pact.extend(action['ang_vel'])
            else: 
		pact.extend(np.zeros(3))
	    # pact[7:25] : actions
	    if 'actions' in action:
		# fill object actions vector
		object_actions = []
		for objact in action['actions']:
		    if 'force' in objact:
			object_actions.extend(objact['force'])
		    else:
			object_actions.extend(np.zeros(3))
		    if 'torque' in objact:
			object_actions.extend(objact['torque'])
		    else:
			object_actions.extend(np.zeros(3))
		    """
			The chosen object not necessarily the one acted upon
			depending on action_pos. The actual object acted upon
			is stored in 'id'
		    """
		    if 'object' in objact:
			object_actions.append(int(objact['object']))
		    else:
			object_actions.append(0)
		    if 'action_pos' in objact:
			action_pos = objact['action_pos']
			if self.crop_size is not None:
			    if self.orig_shape is None:
			        raise IndexError('postproc_img() \
					must be called before postproc_actions()')
			    action_pos[0] = int(action_pos[0] / \
				float(self.orig_shape[0]) * self.crop_size[0])
			    action_pos[1] = int(action_pos[1] / \
				float(self.orig_shape[1]) * self.crop_size[1])
			object_actions.extend(action_pos)
		    else:
			object_actions.extend(np.zeros(2))
		""" 
			Each object action vector has a length of 3+3+1+2=9.
			Object actions are performed on maximally 2 objects
			simultanously (CRASHING action). Thus, the vector length
			has to be 2*9=18
		"""
		while len(object_actions) < 18:
		    object_actions.append(0)
		# append object actions vector
		pact.extend(object_actions)
	    parsed_actions.append(pact)
	return np.array(parsed_actions)

    def next(self):
	batch = super(FuturePredictionData, self).next()
	# create present-future image/action pairs
	img, act, fut_img, fut_act, ids, fut_ids = self.create_image_pairs(batch[self.images], batch[self.actions])
	
	feed_dict = {'images': np.squeeze(img),
		     'actions': np.squeeze(act),
		     'future_images': np.squeeze(fut_img),
		     'future_actions': np.squeeze(fut_act)[:,0].astype(np.int32),
		     'ids': np.squeeze(ids),
		     'future_ids': np.squeeze(fut_ids)}
	return feed_dict

    def create_image_pairs(self, input_images, input_actions):
	"""
	    create present-future image/action pairs with either
		- fixed time differences or 
		- variable time differences 
	    between the image pairs as specified by the user
	"""
	images = []
	actions = []
	future_images = []
	future_actions = []
	ids = []
	future_ids = []
	if len(input_images) < 1 or len(input_actions) < 1:
	    return [images, actions, future_images, future_actions, ids, future_ids]
	
	# specify the length of the action sequence based on the maximally possible delta_t
	delta_t = self.min_time_difference
	if self.random_time:
	    delta_t = self.max_time_difference
	action_sequence_length = delta_t * len(input_actions[0])
	image_sequence_length = delta_t * input_images[0].shape[2] 
	
	# create present-future image/action pairs
	for i in range(len(input_images)):	    
	    # select time delta
	    if self.random_time:
		max_time_difference = min(len(input_images) - i, self.max_time_difference)
		if max_time_difference <= self.min_time_difference \
			or max_time_difference < 1:
		    continue
		delta_t = np.random.randint(self.min_time_difference, max_time_difference)
	    else:
		if i + delta_t >= len(input_images):
		    break
	    images.append(input_images[i])
	    actions.append(input_actions[i])
	    future_images.append(input_images[i+delta_t])
	    future_actions.append(input_actions[i+delta_t])
	    ids.append(i+self.batch_num*self.batch_size)
	    future_ids.append(i+delta_t+self.batch_num*self.batch_size)
