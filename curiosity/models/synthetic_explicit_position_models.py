'''
Some models for that synthetic explicit position data.
'''

import numpy as np
import tensorflow as tf


from curiosity.models.model_building_blocks import ConvNetwithBypasses


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




def not_quite_identity_net(inputs, t_in = 3, t_out = 3, skip = 3, **kwargs):
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
	return tf.nn.l2_loss(pred - diff)
