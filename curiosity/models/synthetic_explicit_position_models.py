'''
Some models for that synthetic explicit position data.
'''

import numpy as np
import tensorflow as tf


from curiosity.models.model_building_blocks import ConvNetwithBypasses


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



def to_old_inputs(inputs):
	X = inputs['X']
	Y = inputs['Y']
	Z = inputs['Z']
	old_list = []
	for t in range(X.get_shape().as_list()[1]):
		X_t = tf.slice(X, [0,t], [-1, 1])
		Y_t = tf.slice(Y, [0,t], [-1, 1])
		Z_t = tf.slice(Z, [0,t], [-1, 1])
		old_list.extend([X_t, Y_t, Z_t])
	return {'positions' : tf.concat(1, old_list)}


def mlp_nonseparate_input(inputs, cfg, stddev = .01, **kwargs):
	m = ConvNetwithBypasses(**kwargs)


	current_node = inputs['in']
	future_node = inputs['out']
	skip_node = inputs['skip']

	print(current_node)
	num_end_features = future_node.get_shape().as_list()[1]
	m.output =  tf.concat(1, [current_node, skip_node])
	print(m.output)
	hidden_loop_with_bypasses(m.output, m, cfg, activation = 'relu', stddev = stddev)
	print(m.output)
	with tf.variable_scope('out'):
		m.fc(num_end_features, init = 'trunc_norm', activation = None, bias = .01, dropout = None)


	print(m.output)

	return {'pred' : m.output, 'tv' : future_node, 'skip' : skip_node, 'in_pos' : current_node}, m.params

def lin_square_lin(inputs, cfg, stddev = .01, **kwargs):
	m = ConvNetwithBypasses(**kwargs)

	current_node = inputs['in']
	future_node = inputs['out']
	skip_node = inputs['skip']

	print(current_node)
	m.output = tf.concat(1, [current_node, skip_node])
	print(m.output)

	num_first_layer = cfg['first_lin']
	with tf.variable_scope('first_lin'):
		m.fc(num_first_layer, init = 'trunc_norm', activation = None, bias = .01, dropout = None, stddev = stddev)

	#now square it!
	m.output = tf.concat(1, [m.output, m.output * m.output])
	print(m.output)

	num_end_features = future_node.get_shape().as_list()[1]
	with tf.variable_scope('out'):
		m.fc(num_end_features, init = 'trunc_norm', activation = None, bias = .01, dropout = None, stddev = stddev)

	print(m.output)

	return {'pred' : m.output, 'tv' : future_node, 'skip' : skip_node, 'in_pos' : current_node}, m.params


def linear_net_nonseparate_input(inputs, **kwargs):
	m = ConvNetwithBypasses(**kwargs)


	current_node = inputs['in']
	future_node = inputs['out']
	skip_node = inputs['skip']

	m.output = current_node
	print(m.output)
	print(future_node)
	print(skip_node)
	num_end_features = future_node.get_shape().as_list()[1]
	with tf.variable_scope('out'):
		m.fc(num_end_features, init = 'trunc_norm', activation = None, bias = .01, dropout = None)

	print(m.output)

	return {'pred' : m.output, 'tv' : future_node, 'skip' : skip_node, 'in_pos' : current_node}, m.params



def linear_net(inputs, t_in = 3, t_out = 3, skip = 3, **kwargs):
	#Just spits back first frames

	m = ConvNetwithBypasses(**kwargs)

	old_inputs = to_old_inputs(inputs)
	current_node = old_inputs['positions'][:, : 3 * t_in]
	future_node = old_inputs['positions'][:, -3 * t_out : ]
	m.output = current_node

	num_end_features = t_out * 3
	with tf.variable_scope('out'):
		m.fc(num_end_features, init = 'trunc_norm', activation = None, bias = .01, dropout = None)

	return {'pred' : m.output, 'tv' : future_node, 'in_pos' : current_node}, m.params

def separate_linear(inputs, t_in = 3, t_out = 3, **kwargs):
	init_vals = []
	final_vals = []
	for desc in ['X', 'Y', 'Z']:
		init_vals.append((desc, inputs[desc][:, :t_in]))
		final_vals.append((desc, inputs[desc][:, -t_out : ]))

	m = ConvNetwithBypasses(**kwargs)
	pred = {}

	for (desc, node) in init_vals:
		with tf.variable_scope('lin_' + desc):
			m.output = node
			pred[desc] = m.fc(t_out, init = 'trunc_norm', activation = None, bias = .01, dropout = None)

	return {'pred' : pred, 'init' : init_vals, 'fin' : final_vals}, m.params



def l2_diff_loss_separate_format(outputs, positions):
	loss_list = []
	for desc in ['X', 'Y', 'Z']:
		diffs = compute_diffs(outputs['init'][desc][:, -1:], outputs['fin'][desc])
		loss_list.append(tf.nn.l2_loss(outputs['pred'][desc] - diffs))
	return sum(loss_list)




def compute_diffs(last_known_positions, subsequent_positions, t_out):
	curr_pos = last_known_positions
	diffs = []
	for i in range(t_out):
		next_pos = subsequent_positions[:, i * 3 : (i + 1) * 3]
		diffs.append(next_pos - curr_pos)
		curr_pos = next_pos
	return tf.concat(1, diffs)

def l2_diff_loss_fn_w_skip(outputs, positions_parsed, t_out = 3):
	pred = outputs['pred']
	tv = outputs['tv']
	in_pos = outputs['in_pos']
	last_positions = in_pos[:, - 3 :]
	diff = compute_diffs(last_positions, tv, t_out)
	return tf.nn.l2_loss(pred - diff) / (3 * t_out)
