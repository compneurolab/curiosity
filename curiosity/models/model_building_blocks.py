'''
Extending tfutils.model Convnet class to have some additional moves.
Maybe move into tfutils.model?
'''
import sys
sys.path.append('tfutils')
import tensorflow as tf
from collections import OrderedDict


from tfutils import model

class ConvNetwithBypasses(model.ConvNet):
	'''Right now just borrowing from chengxuz's contribution...will edit if modified

	See https://github.com/neuroailab/barrel/blob/master/normals_relat/normal_pred/normal_encoder_asymmetric_with_bypass.py'''
	def __init__(self, seed=None, **kwargs):
		super(ConvNetwithBypasses, self).__init__(seed=seed, **kwargs)

	@property
	def params(self):
	    return self._params

	@params.setter
	def params(self, value):
		'''Modified from parent to allow for multiple calls of the same type within a scope name.

		This should not happen unless we are pushing more than one node through the same graph, this keeps a record of that.'''
		name = tf.get_variable_scope().name
		if name not in self._params:
		    self._params[name] = OrderedDict()
		if value['type'] in self._params[name]:
			print('Type reused in scope, should be a reuse of same subgraph!')
			self._params[name][value['type']]['input'] = self._params[name][value['type']]['input'] + ',' + value['input']
		else:
			self._params[name][value['type']] = value


	@tf.contrib.framework.add_arg_scope
	def conv_given_filters(self, kernel, biases, stride = 1, padding = 'SAME', activation = 'relu', batch_normalize = False, in_layer = None):
		if in_layer is None:
			in_layer = self.output
		k_shape = kernel.get_shape().as_list()
		out_shape = k_shape[3]
		ksize1 = k_shape[0]
		ksize2 = k_shape[1]
		conv = tf.nn.conv2d(in_layer, kernel,
		                    strides=[1, stride, stride, 1],
		                    padding=padding)
		
		if batch_normalize:
			#Using "global normalization," which is recommended in the original paper
			print('doing batch normalization')
			mean, var = tf.nn.moments(conv, [0, 1, 2])
			scale = tf.get_variable(initializer=tf.constant_initializer(bias),
			                         shape=[out_shape],
			                         dtype=tf.float32,
			                         name='scale')
			self.output = tf.nn.batch_normalization(conv, mean, var, biases, scale, 1e-3, name = 'conv')
		else:
			self.output = tf.nn.bias_add(conv, biases, name='conv')
		
		if activation is not None:
		    self.output = self.activation(kind=activation)
		
		self.params = {'input': in_layer.name,
		               'type': 'conv',
		               'num_filters': out_shape,
		               'stride': stride,
		               'kernel_size': (ksize1, ksize2),
		               'padding': padding,
		               'init': init,
		               'activation': activation}
		return self.output




	@tf.contrib.framework.add_arg_scope
	def conv(self, 
                 out_shape, 
                 ksize=3, 
                 stride=1, 
                 padding='SAME', 
                 init='xavier', 
                 stddev=.01, 
                 bias=1, 
                 activation='relu', 
                 weight_decay=None, 
                 in_layer=None, 
                 init_file=None, 
                 init_layer_keys=None, 
                 batch_normalize=False,
                 group=None,
                 trainable=True,
                   ):
		if in_layer is None:
		    in_layer = self.output
		if weight_decay is None:
		    weight_decay = 0.
		in_shape = in_layer.get_shape().as_list()[-1]

		if isinstance(ksize, int):
		    ksize1 = ksize
		    ksize2 = ksize
		else:
		    ksize1, ksize2 = ksize

		if init != 'from_file':
		    kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
		                             shape=[ksize1, ksize2, in_shape, out_shape],
		                             dtype=tf.float32,
		                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
		                             name='weights', trainable=trainable)
		    biases = tf.get_variable(initializer=tf.constant_initializer(bias),
		                             shape=[out_shape],
		                             dtype=tf.float32,
		                             name='bias', trainable=trainable)
		else:
		    init_dict = self.initializer(init,
		                                 init_file=init_file,
		                                 init_keys=init_layer_keys)
		    kernel = tf.get_variable(initializer=init_dict['weight'],
		                             dtype=tf.float32,
		                             regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
		                             name='weights', trainable=trainable)
		    biases = tf.get_variable(initializer=init_dict['bias'],
		                             dtype=tf.float32,
		                             name='bias', trainable=trainable)

                if group is None or group == 1:
		    conv = tf.nn.conv2d(in_layer, kernel,
		                    strides=[1, stride, stride, 1],
		                    padding=padding)
                else:
                    print('Partially convolving; group=%d' % group)
                    assert in_layer.get_shape()[-1] % group == 0
                    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride, stride, 1],\
                                                         padding=padding)
                    in_layers = tf.split(3, group, in_layer)
                    kernels = tf.split(3, group, kernel)
                    convs = [convolve(i, k) for i,k in zip(in_layers, kernels)]
                    conv = tf.concat(3, convs)

		if batch_normalize:
			#Using "global normalization," which is recommended in the original paper
			print('doing batch normalization')
			mean, var = tf.nn.moments(conv, [0, 1, 2])
			scale = tf.get_variable(initializer=tf.constant_initializer(bias),
			                         shape=[out_shape],
			                         dtype=tf.float32,
			                         name='scale', trainable=trainable)
			self.output = tf.nn.batch_normalization(conv, mean, var, biases, scale, 1e-3, name = 'conv')
		else:
			self.output = tf.nn.bias_add(conv, biases, name='conv')

		if activation is not None:
		    self.output = self.activation(kind=activation)
		self.params = {'input': in_layer.name,
		               'type': 'conv',
		               'num_filters': out_shape,
		               'stride': stride,
		               'kernel_size': (ksize1, ksize2),
		               'padding': padding,
		               'init': init,
		               'stddev': stddev,
		               'bias': bias,
		               'activation': activation,
		               'weight_decay': weight_decay,
		               'seed': self.seed}
		return self.output


	@tf.contrib.framework.add_arg_scope
	def pool(self,
	         ksize=3,
	         stride=2,
	         padding='SAME',
	         in_layer=None,
	         pfunc='maxpool'):
	    if in_layer is None:
	        in_layer = self.output

	    if isinstance(ksize, int):
	        ksize1 = ksize
	        ksize2 = ksize
	    else:
	        ksize1, ksize2 = ksize

	    if pfunc=='maxpool':
	        self.output = tf.nn.max_pool(in_layer,
	                                     ksize=[1, ksize1, ksize2, 1],
	                                     strides=[1, stride, stride, 1],
	                                     padding=padding,
	                                     name='pool')
	    else:
	        self.output = tf.nn.avg_pool(in_layer,
	                                     ksize=[1, ksize1, ksize2, 1],
	                                     strides=[1, stride, stride, 1],
	                                     padding=padding,
	                                     name='pool')
	    self.params = {'input': in_layer.name,
	                   'type': pfunc,
	                   'kernel_size': (ksize1, ksize2),
	                   'stride': stride,
	                   'padding': padding}
	    return self.output

	def reshape(self, new_size, in_layer=None):
		#TODO: add params update
	    if in_layer is None:
	        in_layer = self.output

	    size_l = [in_layer.get_shape().as_list()[0]]
	    size_l.extend(new_size)
	    self.output = tf.reshape(in_layer, size_l)
	    self.params = {'input' : in_layer.name, 'type' : 'reshape', 'new_shape' : size_l}
	    return self.output

	def resize_images(self, new_size, in_layer=None):
		#TODO: add params update
	    if in_layer is None:
	        in_layer = self.output
	    self.output = tf.image.resize_images(in_layer, [new_size, new_size])
	    self.params = {'input' : in_layer.name, 'type' : 'resize', 'new_size' : new_size}
	    return self.output

	def minmax(self, min_arg = 'inf', max_arg = 'ninf', in_layer = None):
		'''Note that this does nothing, silently, except modify params, if this is called with default arguments.

		'''
		if in_layer is None:
			in_layer = self.output
		self.params = {'input' : in_layer.name, 'type' : 'minmax', 'min_arg' : min_arg, 'max_arg' : max_arg}
		if max_arg != 'ninf':
			in_layer = tf.maximum(in_layer, max_arg)
		if min_arg != 'inf':
			in_layer = tf.minimum(in_layer, min_arg)
		self.output = in_layer
		return self.output


	def add_bypass(self, bypass_layers, in_layer=None):
	    if in_layer is None:
	        in_layer = self.output

	    if not isinstance(bypass_layers, list):
	    	bypass_layers = [bypass_layers]
	    in_shape = in_layer.get_shape().as_list()
	    toconcat = [in_layer]
	    concat_type = None
	    if len(in_shape) == 4:	
	    	ds = in_shape[1]
	    	for layer in bypass_layers:
	    		if layer.get_shape().as_list()[1] != ds:
	    			toconcat.append(self.resize_images(ds, in_layer = layer))
	    		else:
	    			toconcat.append(layer)
	    	self.output = tf.concat(3, toconcat)
	    	concat_type = 'image'
	    elif len(in_shape) == 2:
	    	toconcat.extend(bypass_layers)
	    	self.output = tf.concat(1, toconcat)
	    	concat_type = 'flat'
	    else:
	    	raise Exception('Bypass case not yet handled.')
	    self.params = {'input' : in_layer.name, 'type' : 'bypass', 'bypass_names' : [l.name for l in toconcat[1:]], 'concat_type' : concat_type}
	    return self.output
