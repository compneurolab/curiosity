'''
Asymmetric architectures, a bit more modularized. No more slippage implemented.
Takes a cfg passed in form params.
'''

import numpy as np
import tensorflow as tf
import zmq

from curiosity.models.model_building_blocks import ConvNetwithBypasses

def feedforward_conv_loop(input_node, m, cfg, reuse_weights = False):
	m.output = input_node
	encode_nodes = [input_node]
	#encoding
	encode_depth = cfg['encode_depth']
	print('Encode depth: %d' % encode_depth)
	cfs0 = None

	for i in range(1, encode_depth + 1):
	#not sure this usage ConvNet class creates exactly the params that we want to have, specifically in the 'input' field, but should give us an accurate record of this network's configuration
		with tf.variable_scope('encode' + str(i)) as scope:
			if reuse_weights:
				scope.reuse_variables()

			with tf.contrib.framework.arg_scope([m.conv], init='trunc_norm', stddev=.01, bias=0, activation='relu'):
			    cfs = cfg['encode'][i]['filter_size']
			    cfs0 = cfs
			    nf = cfg['encode'][i]['num_filters']
			    cs = cfg['encode'][i]['stride']
			    m.conv(nf, cfs, cs)
	#TODO add print function
			pool = cfg['encode'][i].get('pool')
			if pool:
			    pfs = pool['size']
			    ps = pool['stride']
			    m.pool(pfs, ps)
			encode_nodes.append(m.output)
	return encode_nodes

def hidden_loop(input_node, m, cfg, reuse_weights = False):
	assert len(input_node.get_shape().as_list()) == 2, len(input_node.get_shape().as_list())
	hidden_depth = cfg['hidden_depth']
	m.output = input_node
	for i in range(1, hidden_depth + 1):
		with tf.variable_scope('hidden' + str(i)) as scope:
			if reuse_weights:
				scope.reuse_variables()
			nf = cfg['hidden'][i]['num_features']
			m.fc(nf, init = 'trunc_norm', activation = 'relu', bias = .01, dropout = None)
	return m.output

def decode_conv_loop(input_node, m, cfg, nodes_for_bypass, reuse_weights = False):
	assert len(input_node, get_shape().as_list()) == 4, len(input_node, get_shape().as_list())
	m.output = input_node
	decode_depth = cfg['decode_depth']
	for i in range(1, decode_depth + 1):
		with tf.variable_scope('decode' + str(i)) as scope:
			if reuse_weights:
				scope.reuse_variables()
			ds = cfg['decode'][i]['size']
			m.resize_images(ds)
			bypass = cfg['decode'][i].get('bypass')
			if bypass:
				bypass_layer = nodes_for_bypass[bypass]
				m.add_bypass(bypass_layer)
			nf1 = cfg['decode'][i]['num_filters']
			cfs = cfg['decode'][i]['filter_size']
			if i < decode_depth:
				m.conv(nf1, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation='relu')
			else:
				m.conv(nf1, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation=None)


def std_asymmetric_model(inputs, cfg = None, num_channels = 3, T_in = 1, T_out = 1, rng = None **kwargs):
	actions_sequence = tf.cast(inputs['parsed_actions'], tf.float32)
	current_node = inputs['images'][:, :, :, :num_channels * T_in]
	current_node = tf.divide(tf.cast(current_node, tf.float32), 255.)
	future_node = inputs['images'][:, :, :, num_channels * T_in : ]
	assert num_channels * (T_in + T_out) == inputs['images'].get_shape().as_list()[3]

	if rng is None:
		rng = np.random.RandomState(seed=kwargs['seed'])

	m = ConvNetwithBypasses(**kwargs)

	#encode
	encode_nodes = feedforward_conv_loop(current_node, m, cfg, reuse_weights = False)
	
	#flatten
	enc_shape = m.output.get_shape().as_list()
	m.reshape([np.prod(enc_shape[1:])])

	#hidden
	hidden_loop(m.output, m, cfg, reuse_weights = False)

	#unflatten
	ds = cfg['decode'][0]['size']
	nf1 = cfg['decode'][0]['num_filters']
	m.reshape([ds, ds, nf1])

	decode_conv_loop(m.output, m, cfg, encode_nodes, reuse_weights = False)

	return {'pred' : m.output, 'tv' : future_node}, m.params


def something_or_nothing_loss_fn(outputs, image, threshold = None, num_channels = 3, **kwargs):
	print('inside loss')
	print(outputs)
	print(image)
	pred = outputs['pred']
	future_images = tf.cast(outputs['tv'], 'float32')
	assert threshold is not None
	T_in = int((image.get_shape().as_list()[-1] -  pred.get_shape().as_list()[-1]) / num_channels)
	original_image = image[:, :, :, (T_in - 1) * num_channels: T_in * num_channels]
	original_image = tf.cast(original_image, 'float32')
	diffs = compute_diffs_timestep_1(original_image, future_images, num_channels = num_channels)
	#just measure some absolute change relative to a threshold
	diffs = tf.abs(diffs / 255.) - threshold
	tv = tf.cast(tf.ceil(diffs), 'uint8')
	tv = tf.one_hot(tv, depth = 2)
	my_shape = pred.get_shape().as_list()
	my_shape.append(1)
	pred = tf.reshape(pred, my_shape)
	pred = tf.concat(4, [tf.zeros(my_shape), pred])
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, tv))





