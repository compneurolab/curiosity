'''
Type (2) model definitions.
Normals +_2 Objects +_2 Actions -> objects
'''

import explicit_future_prediction_base as fp_base
import tensorflow as tf
from curiosity.models.model_building_blocks import ConvNetwithBypasses
import numpy as np


def feedforward_conv_loop(input_node, m, cfg, bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False):
	m.output = input_node
	encode_nodes = [input_node]
	#encoding
	encode_depth = cfg['encode_depth']
	print('Encode depth: %d' % encode_depth)
	cfs0 = None

	if bypass_nodes is None:
		bypass_nodes = [m.output]

	for i in range(1, encode_depth + 1):
	#not sure this usage ConvNet class creates exactly the params that we want to have, specifically in the 'input' field, but should give us an accurate record of this network's configuration
		with tf.variable_scope('encode' + str(i)) as scope:
			if reuse_weights:
				scope.reuse_variables()

			bypass = cfg['encode'][i].get('bypass')
			if bypass:
				if type(bypass) == list:
					bypass_node = [bypass_nodes[bp] for bp in bypass]
				else:
					bypass_node = bypass_nodes[bypass]
				m.add_bypass(bypass_node)

			bn = cfg['encode'][i]['conv'].get('batch_normalize')
			if bn:
				norm_it = bn
			else:
				norm_it = batch_normalize



			with tf.contrib.framework.arg_scope([m.conv], init='trunc_norm', stddev=.01, bias=0, batch_normalize = norm_it):
			    cfs = cfg['encode'][i]['conv']['filter_size']
			    cfs0 = cfs
			    nf = cfg['encode'][i]['conv']['num_filters']
			    cs = cfg['encode'][i]['conv']['stride']
			    print('conv shape to shape')
			    print(m.output)
			    if no_nonlinearity_end and i == encode_depth:
			    	m.conv(nf, cfs, cs, activation = None)
			    else:
			    	my_activation = cfg['encode'][i].get('nonlinearity')
			    	if my_activation is None:
			    		my_activation = 'relu'
			    	m.conv(nf, cfs, cs, activation = my_activation)
			    print(m.output)
	#TODO add print function
			pool = cfg['encode'][i].get('pool')
			if pool:
			    pfs = pool['size']
			    ps = pool['stride']
			    m.pool(pfs, ps)
			encode_nodes.append(m.output)
			bypass_nodes.append(m.output)
	return encode_nodes


def hidden_loop_with_bypasses(input_node, m, cfg, nodes_for_bypass = [], stddev = .01, reuse_weights = False, activation = 'relu'):
	assert len(input_node.get_shape().as_list()) == 2, len(input_node.get_shape().as_list())
	hidden_depth = cfg['hidden_depth']
	m.output = input_node
	print('in hidden loop')
	print(m.output)
	for i in range(1, hidden_depth + 1):
		with tf.variable_scope('hidden' + str(i)) as scope:
			if reuse_weights:
				scope.reuse_variables()
			bypass = cfg['hidden'][i].get('bypass')
			if bypass:
				bypass_node = nodes_for_bypass[bypass]
				m.add_bypass(bypass_node)
			nf = cfg['hidden'][i]['num_features']
			m.fc(nf, init = 'trunc_norm', activation = activation, bias = .01, stddev = stddev, dropout = None)
			nodes_for_bypass.append(m.output)
			print(m.output)
	return m.output



def simple_conv_to_mlp_structure(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, **kwargs):
	base_net = fp_base.FuturePredictionBaseModel(inputs, time_seen, normalization_method = normalization_method, stats_file = stats_file)
	to_concat_attributes_float32 = ['normals', 'normals2', 'object_data_seen', 'actions_seen', 'actions_future']
	to_concat = []
	for attr in to_concat_attributes_float32:
		to_concat.extend(tf.unstack(base_net.inputs[attr], axis = 1))
	in_concat = tf.concat(to_concat, axis = 3)

	m = ConvNetwithBypasses(**kwargs)
	#encode
	encode_nodes = feedforward_conv_loop(in_concat, m, cfg, reuse_weights = False)

	enc_shape = m.output.get_shape().as_list()
	m.reshape([np.prod(enc_shape[1:])])

	#hidden
	pred = hidden_loop_with_bypasses(m.output, m, cfg, reuse_weights = False)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params

def l2_loss(outputs):
	pred =  outputs['pred']
	tv = outputs['object_data_future']
	tv = tf.reshape(tv, [tv.get_shape().as_list()[0], -1])
	n_entries = tv.get_shape().as_list()[1] * tv.get_shape().as_list()[0]
	return tf.nn.l2_loss(pred - tv) / n_entries

def compute_diffs(last_seen_data, future_data):
	diffed_data_list = []
	for t in range(future_data.get_shape().as_list()[1]):
		diffed_data_list.append(future_data[:, t] - last_seen_data)
		last_seen_data = future_data[:, t]
	return tf.concat(diffed_data_list, axis = 1)

def l2_diff_loss(outputs):
	pred = outputs['pred']
	future_dat = outputs['object_data_future']
	seen_dat = outputs['object_data_seen_1d']
	last_seen_dat = seen_dat[:, -1]
	tv = compute_diffs(last_seen_dat, future_dat)
	tv = tf.reshape(tv, [tv.get_shape().as_list()[0], -1])
	n_entries = tv.get_shape().as_list()[1] * tv.get_shape().as_list()[0]
	return tf.nn.l2_loss(pred - tv) / n_entries


cfg_simple = {
	'encode_depth' : 6,
	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 32}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 32}},
		3 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 32}},
		4 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 16}},
		5 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 16}},
		6 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 16}} 
	},

	'hidden_depth' : 2,
	'hidden' : {
		1 : {'num_features' : 60},
		2 : {'num_features' : 60} #2 points * 5 timesteps * (2 + 4) dimension
	}
}








