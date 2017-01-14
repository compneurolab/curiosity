'''
Extending tfutils.model Convnet class to have some additional moves.
Maybe move into tfutils.model?
'''
import sys
sys.path.append('tfutils')
import tensorflow as tf

from tfutils import model

class ConvNetwithBypasses(model.ConvNet):
	'''Right now just borrowing from chengxuz's contribution...will edit if modified

	See https://github.com/neuroailab/barrel/blob/master/normals_relat/normal_pred/normal_encoder_asymmetric_with_bypass.py'''
	def __init__(self, seed=None, **kwargs):
		super(ConvNetwithBypasses, self).__init__(seed=seed, **kwargs)

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
	    return self.output

	def resize_images(self, new_size, in_layer=None):
		#TODO: add params update
	    if in_layer is None:
	        in_layer = self.output
	    self.output = tf.image.resize_images(in_layer, [new_size, new_size])
	    return self.output

	def add_bypass(self, bypass_layer, in_layer=None):
		#TODO add params update
		#TODO add multiple concatenations
		#not sure best way to do this...doesn't contain where the bypass layer comes from, as is
	    if in_layer is None:
	        in_layer = self.output

	    bypass_shape = bypass_layer.get_shape().as_list()
	    ds = in_layer.get_shape().as_list()[1]
	    if bypass_shape[1] != ds:
	        bypass_layer = tf.image.resize_images(bypass_layer, [ds, ds])
	    self.output = tf.concat(3, [in_layer, bypass_layer])

	    return self.output
