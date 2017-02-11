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




