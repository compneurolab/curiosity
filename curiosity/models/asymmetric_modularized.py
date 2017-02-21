'''
Asymmetric architectures, a bit more modularized. No more slippage implemented.
Takes a cfg passed in form params.
'''

import numpy as np
import tensorflow as tf


from curiosity.models.model_building_blocks import ConvNetwithBypasses

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
			    	m.conv(nf, cfs, cs, activation='relu')
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

def hidden_loop_with_bypasses(input_node, m, cfg, nodes_for_bypass, reuse_weights = False):
	assert len(input_node.get_shape().as_list()) == 2, len(input_node.get_shape().as_list())
	hidden_depth = cfg['hidden_depth']
	m.output = input_node
	for i in range(1, hidden_depth + 1):
		with tf.variable_scope('hidden' + str(i)) as scope:
			if reuse_weights:
				scope.reuse_variables()
			bypass = cfg['hidden'][i].get('bypass')
			if bypass:
				bypass_node = nodes_for_bypass[bypass]
				m.add_bypass(bypass_node)
			nf = cfg['hidden'][i]['num_features']
			m.fc(nf, init = 'trunc_norm', activation = 'relu', bias = .01, dropout = None)
			nodes_for_bypass.append(m.output)
			print(m.output)
	return m.output

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
			print(m.output)
	return m.output

def decode_conv_loop(input_node, m, cfg, nodes_for_bypass, reuse_weights = False, batch_normalize = False):
	assert len(input_node.get_shape().as_list()) == 4, len(input_node.get_shape().as_list())
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
				if type(bypass) == list:
					bypass_node = [nodes_for_bypass[bp] for bp in bypass]
				else:
					bypass_node = nodes_for_bypass[bypass]
				m.add_bypass(bypass_node)
			nf1 = cfg['decode'][i]['num_filters']
			cfs = cfg['decode'][i]['filter_size']
			print('decode dims')
			print(m.output.get_shape().as_list())

			bn = cfg['decode'][i].get('batch_normalize')
			if bn:
				norm_it = bn
			else:
				norm_it = batch_normalize


			if i < decode_depth:
				m.conv(nf1, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation='relu', batch_normalize = norm_it)
			else:
				m.conv(nf1, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation=None, batch_normalize = False) #assuming we don't want to batch normalize at the end
			print(m.output.get_shape().as_list())
			nodes_for_bypass.append(m.output)
	return m.output




def std_asymmetric_model(inputs, cfg = None, num_channels = 3, T_in = 1, T_out = 1, rng = None, **kwargs):
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
	m.add_bypass(actions_sequence)

	#hidden
	hidden_loop(m.output, m, cfg, reuse_weights = False)

	#unflatten
	ds = cfg['decode'][0]['size']
	nf1 = cfg['decode'][0]['num_filters']
	m.reshape([ds, ds, nf1])

	decode_conv_loop(m.output, m, cfg, encode_nodes, reuse_weights = False)

	return {'pred' : m.output, 'tv' : future_node}, m.params

def somewhat_less_simple_net(inputs, skip = 1, cfg = None, num_channels = 3, T_in = 3, T_out = 3, batch_normalize = False, **kwargs):
	actions_sequence = tf.cast(inputs['parsed_actions'], tf.float32)
	current_node = inputs['images'][:, :, :, :num_channels * T_in]
	current_node = tf.divide(tf.cast(current_node, tf.float32), 255.)
	future_node1 = inputs['images'][:, :, :, num_channels * T_in : ]
	future_node2 = inputs['future_images']
	future_node = tf.concat(3, [future_node1, future_node2])
	if skip > 1:
		print('lengths')
		c_list = [current_node[:, :, :, num_channels * i : num_channels * (i + 1)] for i in range(0, T_in, skip)]
		print(len(c_list))
		current_node = tf.concat(3, c_list)
		print(current_node.get_shape().as_list())
		f_list = [future_node[:, :, :, num_channels * i : num_channels * (i + 1)] for i in range(skip - 1, T_out, skip)]
		print(len(f_list))
		future_node = tf.concat(3, f_list)
		print(future_node.get_shape().as_list())
	assert num_channels * (T_in + T_out) == inputs['images'].get_shape().as_list()[3] + num_channels

	m = ConvNetwithBypasses(**kwargs)

	encode_nodes = feedforward_conv_loop(current_node, m, cfg, reuse_weights = False, batch_normalize = batch_normalize)

	decode_conv_loop(m.output, m, cfg, encode_nodes, reuse_weights = False)

	return {'pred' : m.output, 'tv' : future_node}, m.params


def simple_net(inputs, cfg = None, num_channels = 3, T_in = 3, T_out = 3, batch_normalize = False, **kwargs):
	actions_sequence = tf.cast(inputs['parsed_actions'], tf.float32)
	current_node = inputs['images'][:, :, :, :num_channels * T_in]
	current_node = tf.divide(tf.cast(current_node, tf.float32), 255.)
	future_node = inputs['images'][:, :, :, num_channels * T_in : ]
	assert num_channels * (T_in + T_out) == inputs['images'].get_shape().as_list()[3]

	m = ConvNetwithBypasses(**kwargs)

	feedforward_conv_loop(current_node, m, cfg, reuse_weights = False, batch_normalize = batch_normalize, no_nonlinearity_end = True)

	return {'pred' : m.output, 'tv' : future_node}, m.params

def nothing_net(inputs, **kwargs):
	future_node = inputs['images'][:, :, :, num_channels * T_in : ]
	future_shape = future_node.get_shape().as_list()
	trivial_ans = tf.zeros(future_shape)

	m = ConvNetwithBypasses(**kwargs)

	return {'pred' : trivial_ans, 'tv' : future_node}, m.params


def asymmetric_with_bottleneck(inputs, cfg = None, num_channels = 3, T_in = 3, T_out = 3, batch_normalize = False, **kwargs):
	actions_sequence = tf.cast(inputs['parsed_actions'], tf.float32)
	current_node = inputs['images'][:, :, :, :num_channels * T_in]
	current_node = tf.divide(tf.cast(current_node, tf.float32), 255.)
	future_node = inputs['images'][:, :, :, num_channels * T_in : ]
	assert num_channels * (T_in + T_out) == inputs['images'].get_shape().as_list()[3]

	m = ConvNetwithBypasses(**kwargs)

	#encode
	encode_nodes = feedforward_conv_loop(current_node, m, cfg, reuse_weights = False, batch_normalize = batch_normalize)
	
	#flatten
	enc_shape = m.output.get_shape().as_list()
	m.reshape([np.prod(enc_shape[1:])])
	m.add_bypass(actions_sequence)
	nf = cfg['hidden']['to_repn']['num_features']
	m.fc(nf, init = 'trunc_norm', activation = 'relu', bias = .01, dropout = None)
	hidden_depth = cfg['hidden_depth']
	n_dynamic = cfg['hidden'][hidden_depth - 1]['num_features']
	print('num dynamic ' + str(n_dynamic))
	static_node = m.output[:, n_dynamic : ]
	nodes_for_bypass = [static_node]

	#hidden
	hidden_loop_with_bypasses(m.output, m, cfg, nodes_for_bypass)

	#unflatten
	ds = cfg['decode'][0]['size']
	nf1 = cfg['decode'][0]['num_filters']
	m.reshape([ds, ds, nf1])

	decode_conv_loop(m.output, m, cfg, encode_nodes, reuse_weights = False)

	return {'pred' : m.output, 'tv' : future_node}, m.params



def timestep_1_mlp_model(inputs, cfg = None, num_channels = 3, T_in = 3, T_out = 3, rng = None, **kwargs):
	'''Some basic recurrence: the hidden layer is an mlp that "evolves one timestep."

	So it predicts one loop, and the result is fed back in and augmented with another image.
	A problem here: unclear how one should do bypasses, as decode inputs should somehow be timestep-distance independent,
	and we don't get a full image for predictions down the line, if we're discretizing for the loss.
	'''
	actions_sequence = tf.cast(inputs['parsed_actions'], tf.float32)
	current_node = inputs['images'][:, :, :, :num_channels * T_in]
	current_node = tf.divide(tf.cast(current_node, tf.float32), 255.)
	future_node = inputs['images'][:, :, :, num_channels * T_in : ]
	assert num_channels * (T_in + T_out) == inputs['images'].get_shape().as_list()[3]

	#just so I don't forget to check this...
	assert actions_sequence.get_shape().as_list()[1] == 25 * (T_in + T_out), (actions_sequence.get_shape().as_list()[1], T_in, T_out)
	actions_length = int(actions_sequence.get_shape().as_list()[1] / (T_in + T_out))
	current_actions = actions_sequence[:, :actions_length * T_in]
	future_actions = [actions_sequence[:, actions_length * t: actions_length * (t + 1)] for t in range(T_in, T_in + T_out)]

	if rng is None:
		rng = np.random.RandomState(seed=kwargs['seed'])

	m = ConvNetwithBypasses(**kwargs)

	#encode
	encode_nodes = feedforward_conv_loop(current_node, m, cfg, reuse_weights = False)

	#flatten
	enc_shape = m.output.get_shape().as_list()
	m.reshape([np.prod(enc_shape[1:])])
	#include actions up until before prediction step
	m.add_bypass(current_actions)
	#one fc layer getting to "current state"
	nf = cfg['hidden'][0]['num_features']
	m.fc(nf, init = 'trunc_norm', activation = 'relu', bias = .01, dropout = None)
	current_state = m.output

	predictions = []

	for t in range(T_out):
		m.add_bypass(future_actions[t], in_layer = current_state)
		if t == 0:
			reuse_weights = False
		else:
			reuse_weights = True
		next_state = hidden_loop(m.output, m, cfg, reuse_weights = reuse_weights)
		assert current_state.get_shape().as_list() == next_state.get_shape().as_list(), (current_state.get_shape().as_list(), next_state.get_shape().as_list())
		current_state = next_state
		#unflatten
		ds = cfg['decode'][0]['size']
		nf1 = cfg['decode'][0]['num_filters']
		m.reshape([ds, ds, nf1])
		#decode
		future_decoded = decode_conv_loop(m.output, m, cfg, encode_nodes, reuse_weights = reuse_weights)
		predictions.append(future_decoded)

	return {'pred' : tf.concat(3, predictions), 'tv' : future_node}, m.params


def compute_diffs_timestep_1(original_image, subsequent_images, num_channels = 3):
  curr_image = original_image
  diffs = []
  for i in range(int(subsequent_images.get_shape().as_list()[-1] / num_channels)):
    next_image = subsequent_images[:, :, :, num_channels * i : num_channels * (i + 1)]
    diffs.append(next_image - curr_image)
    curr_image = next_image
  return tf.concat(3, diffs)



def something_or_nothing_loss_fn(outputs, image, threshold = None, num_channels = 3, **kwargs):
	print('in loss')
	print(outputs)
	print(image)
	pred = outputs['pred']
	future_images = tf.cast(outputs['tv'], 'float32')
	assert threshold is not None
	original_image = image[:, :, :, T_in * num_channels: T_in * num_channels]
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





