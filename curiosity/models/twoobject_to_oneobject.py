'''
Type (2) model definitions.
Normals +_2 Objects +_2 Actions -> objects
'''

import explicit_future_prediction_base as fp_base
import tensorflow as tf
from curiosity.models.model_building_blocks import ConvNetwithBypasses
import numpy as np



def feedforward_conv_loop(input_node, m, cfg, desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False):
	m.output = input_node
	encode_nodes = [input_node]
	#encoding
	encode_depth = cfg[desc + '_depth']
	print('Encode depth: %d' % encode_depth)
	cfs0 = None

	if bypass_nodes is None:
		bypass_nodes = [m.output]

	for i in range(1, encode_depth + 1):
	#not sure this usage ConvNet class creates exactly the params that we want to have, specifically in the 'input' field, but should give us an accurate record of this network's configuration
		with tf.variable_scope(desc + str(i)) as scope:
			if reuse_weights:
				scope.reuse_variables()

			bypass = cfg[desc][i].get('bypass')
			if bypass:
				if type(bypass) == list:
					bypass_node = [bypass_nodes[bp] for bp in bypass]
				else:
					bypass_node = bypass_nodes[bypass]
				m.add_bypass(bypass_node)

			bn = cfg[desc][i]['conv'].get('batch_normalize')
			if bn:
				norm_it = bn
			else:
				norm_it = batch_normalize



			with tf.contrib.framework.arg_scope([m.conv], init='xavier', stddev=.01, bias=0, batch_normalize = norm_it):
			    cfs = cfg[desc][i]['conv']['filter_size']
			    cfs0 = cfs
			    nf = cfg[desc][i]['conv']['num_filters']
			    cs = cfg[desc][i]['conv']['stride']
			    print('conv shape to shape')
			    print(m.output)
			    if no_nonlinearity_end and i == encode_depth:
			    	m.conv(nf, cfs, cs, activation = None)
			    else:
			    	my_activation = cfg[desc][i].get('nonlinearity')
			    	if my_activation is None:
			    		my_activation = 'relu'
			    	m.conv(nf, cfs, cs, activation = my_activation)
			    print(m.output)
	#TODO add print function
			pool = cfg[desc][i].get('pool')
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
			my_activation = cfg['hidden'][i].get('activation')
			if my_activation is None:
				my_activation = activation
			m.fc(nf, init = 'xavier', activation = my_activation, bias = .01, stddev = stddev, dropout = None)
			nodes_for_bypass.append(m.output)
			print(m.output)
	return m.output


def just_1d_stuff(inputs, cfg = None, time_seen = None, normalization_method = None, 
					stats_file = None, add_gaussians = True, image_height = None, image_width = None, **kwargs):
	base_net = fp_base.FuturePredictionBaseModel(inputs, time_seen, normalization_method = normalization_method, stats_file = stats_file, 
						add_gaussians = add_gaussians, img_height = image_height, img_width = image_width)
	m = ConvNetwithBypasses(**kwargs)
	in_node = base_net.inputs['object_data_seen_1d']
	in_shape = in_node.get_shape().as_list()
	m.output = in_node
	in_node = m.reshape([np.prod(in_shape[1:])])
	act_node = base_net.inputs['actions_no_pos']
	act_shape = act_node.get_shape().as_list()
	m.output = act_node
	act_node = m.reshape([np.prod(act_shape[1:])])
	m.output = tf.concat([in_node, act_node], axis = 1)
	pred = hidden_loop_with_bypasses(m.output, m, cfg, reuse_weights = False)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params

def just_1d_with_agent_data(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, add_gaussians = True, image_height = None, image_width = None, **kwargs):
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, 
					time_seen = time_seen, stats_file = stats_file, add_gaussians = add_gaussians, img_height = image_height,
					img_width = image_width)
	m = ConvNetwithBypasses(**kwargs)
	in_node = base_net.inputs['object_data_seen_1d']
	in_shape = in_node.get_shape().as_list()
	m.output = in_node
	in_node = m.reshape([np.prod(in_shape[1:])])
	act_node = base_net.inputs['actions_no_pos']
	act_shape = act_node.get_shape().as_list()
	m.output = act_node
	act_node = m.reshape([np.prod(act_shape[1:])])
	agent_node = inputs['agent_data'][:, :,  3:]
	agent_node = tf.reshape(agent_node, [agent_node.get_shape().as_list()[0], -1])
	m.output = tf.concat([in_node, act_node, agent_node], axis = 1)
	pred = hidden_loop_with_bypasses(m.output, m, cfg, reuse_weights = False)
	pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
	pred_shape[3] = 2
	pred = tf.reshape(pred, pred_shape)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params

def just_1d_new_provider(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, add_gaussians = True, image_height = None, image_width = None, **kwargs):
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, 
					time_seen = time_seen, stats_file = stats_file, add_gaussians = add_gaussians, img_height = image_height,
					img_width = image_width)
	m = ConvNetwithBypasses(**kwargs)
	in_node = base_net.inputs['object_data_seen_1d']
	in_shape = in_node.get_shape().as_list()
	m.output = in_node
	in_node = m.reshape([np.prod(in_shape[1:])])
	act_node = base_net.inputs['actions_no_pos']
	act_shape = act_node.get_shape().as_list()
	m.output = act_node
	act_node = m.reshape([np.prod(act_shape[1:])])
	m.output = tf.concat([in_node, act_node], axis = 1)
	pred = hidden_loop_with_bypasses(m.output, m, cfg, reuse_weights = False)
	pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
	pred_shape[3] = 2
	pred = tf.reshape(pred, pred_shape)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params

def just_1d_wdepth(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, add_gaussians = True, image_height = None, image_width = None, **kwargs):
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, 
					time_seen = time_seen, stats_file = stats_file, add_gaussians = add_gaussians, img_height = image_height,
					img_width = image_width)
	m = ConvNetwithBypasses(**kwargs)
	in_node = base_net.inputs['object_data_seen_1d']
	in_shape = in_node.get_shape().as_list()
	m.output = in_node
	in_node = m.reshape([np.prod(in_shape[1:])])
	act_node = base_net.inputs['actions_no_pos']
	act_shape = act_node.get_shape().as_list()
	batch_size = act_shape[0]
	m.output = act_node
	act_node = m.reshape([np.prod(act_shape[1:])])
	depth_node = tf.reshape(base_net.inputs['depth_seen'], [batch_size, -1])
	m.output = tf.concat([in_node, act_node, depth_node], axis = 1)
	pred = hidden_loop_with_bypasses(m.output, m, cfg, reuse_weights = False)
	pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
	pred_shape[3] = 2
	pred = tf.reshape(pred, pred_shape)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params


def shared_weight_conv(inputs, cfg = None, normalization_method = None, stats_file = None, image_height = None, image_width = None, **kwargs):
	batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
	long_len = inputs['object_data'].get_shape().as_list()[1]
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file)
	inputs = base_net.inputs
	m = ConvNetwithBypasses(**kwargs)
	conv_input_names = ['normals', 'normals2', 'object_data_seen', 'actions_seen']
	input_per_time = [tf.concat([inputs[nm][:, t] for nm in conv_input_names], axis = 3) for t in range(time_seen)]
	
	#encode! allows for bypasses if we want em
	reuse_weights = False
	encoded_input = []
	for t in range(time_seen):
		encoded_input.append(feedforward_conv_loop(input_per_time[t], m, cfg, bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1])
		reuse_weights = True

	#flatten and concat
	flattened_input = [tf.reshape(enc_in, [batch_size, -1]) for enc_in in encoded_input]
	flattened_input.append(tf.reshape(inputs['object_data_seen_1d'], [batch_size, -1]))
	flattened_input.append(tf.reshape(inputs['actions_no_pos'], [batch_size, -1]))

	assert len(flattened_input[0].get_shape().as_list()) == 2
	concat_input = tf.concat(flattened_input, axis = 1)

	pred = hidden_loop_with_bypasses(concat_input, m, cfg, reuse_weights = False)
	pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
	pred_shape[3] = 2
	pred = tf.reshape(pred, pred_shape)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params

def one_to_two_to_one(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, obj_pic_dims = None, **kwargs):
	batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
	long_len = inputs['object_data'].get_shape().as_list()[1]
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file)
	inputs = base_net.inputs
	m = ConvNetwithBypasses(**kwargs)

	size_1_attributes = ['normals', 'normals2', 'images', 'images2', 'objects', 'objects2']
	flat_inputs = ['object_data_seen_1d', 'actions_no_pos', 'depth_seen']
	size_1_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
	flat_input_per_time = [tf.concat([tf.reshape(inputs[nm][:, t], [batch_size, -1]) for nm in flat_inputs], axis = 1) for t in range(time_seen)]

	encoded_input = []
	reuse_weights = False
	for t in range(time_seen):
		size_1_encoding_before_concat = feedforward_conv_loop(size_1_input_per_time[t], m, cfg, desc = 'size_1_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
		with tf.variable_scope('coord_to_conv') as scope:
			if reuse_weights:
				scope.reuse_variables()
			coord_res = m.coord_to_conv(cfg['coord_to_conv'][0]['out_shape'], flat_input_per_time[t], ksize = 1, activation = cfg['coord_to_conv'][0]['activation'])
		coord_res = feedforward_conv_loop(coord_res, m, cfg, desc = 'coord_to_conv', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1]
		concat_inputs = tf.concat([size_1_encoding_before_concat[-1], coord_res], axis = 3)
		encoded_input.append(feedforward_conv_loop(concat_inputs, m, cfg, desc = 'encode', bypass_nodes = size_1_encoding_before_concat, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1])
		reuse_weights = True

	#flatten and concat
	flattened_input = [tf.reshape(enc_in, [batch_size, -1]) for enc_in in encoded_input]
#	flattened_input.append(tf.reshape(inputs['object_data_seen_1d'], [batch_size, -1]))
#	flattened_input.append(tf.reshape(inputs['actions_no_pos'], [batch_size, -1]))
	flattened_input = flattened_input + flat_input_per_time


	assert len(flattened_input[0].get_shape().as_list()) == 2
	concat_input = tf.concat(flattened_input, axis = 1)

	pred = hidden_loop_with_bypasses(concat_input, m, cfg, reuse_weights = False)
	pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
	pred_shape[3] = 2
	pred = tf.reshape(pred, pred_shape)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params

def one_step_more_data(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, obj_pic_dims = None, scale_down_height = None, scale_down_width = None, add_depth_gaussian = False, include_pose = False, **kwargs):
        batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
        long_len = inputs['object_data'].get_shape().as_list()[1]
        base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file, scale_down_height = scale_down_height, scale_down_width = scale_down_width, add_depth_gaussian = add_depth_gaussian)
        inputs = base_net.inputs

        size_1_attributes = ['normals', 'normals2', 'images']
        size_2_attributes = ['object_data_seen', 'actions_seen']
        size_1_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
        size_2_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_2_attributes], axis = 3) for t in range(time_seen)]
        m = ConvNetwithBypasses(**kwargs)

        encoded_input = []
        reuse_weights = False
        for t in range(time_seen):
                size_1_encoding_before_concat = feedforward_conv_loop(size_1_input_per_time[t], m, cfg, desc = 'size_1_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
                size_2_encoding_before_concat = feedforward_conv_loop(size_2_input_per_time[t], m, cfg, desc = 'size_2_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
                assert size_1_encoding_before_concat[-1].get_shape().as_list()[:-1] == size_2_encoding_before_concat[-1].get_shape().as_list()[:-1], (size_1_encoding_before_concat[-1].get_shape().as_list()[:-1], size_2_encoding_before_concat[-1].get_shape().as_list()[:-1])
                concat_inputs = tf.concat([size_1_encoding_before_concat[-1], size_2_encoding_before_concat[-1]], axis = 3)
                encoded_input.append(feedforward_conv_loop(concat_inputs, m, cfg, desc = 'encode', bypass_nodes = size_1_encoding_before_concat, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1])
                reuse_weights = True

        #flatten and concat
        flattened_input = [tf.reshape(enc_in, [batch_size, -1]) for enc_in in encoded_input]
        flattened_input.append(tf.reshape(inputs['object_data_seen_1d'], [batch_size, -1]))
        flattened_input.append(tf.reshape(inputs['depth_seen'], [batch_size, -1]))

        assert len(flattened_input[0].get_shape().as_list()) == 2
        concat_input = tf.concat(flattened_input, axis = 1)

        act = inputs['actions_no_pos']
        #print('\033[91mHELLO!!!\033[0m')
        concat_input = tf.tile(tf.expand_dims(concat_input, 1),
                                [1, act.get_shape().as_list()[1], 1])
        time_input = tf.concat([concat_input, act], 2)
        time_input = tf.unstack(time_input, axis=1)

        future_time = base_net.inputs['object_data_future'].get_shape().as_list()[1]
        current_time = inputs['object_data_seen_1d'].get_shape().as_list()[1]
        pred = []
        for i in xrange(current_time, current_time+future_time):
            if i == current_time:
                pos = tf.slice(inputs['object_data_seen_1d'], 
                        [0,current_time-1,0,4], [-1,1,-1,-1]) #[b,1,1,feat]
            else:
                pos = pred_pos
            inp = tf.concat([tf.squeeze(pos), time_input[i]], 1)

            if i == current_time:
                pred_pos = hidden_loop_with_bypasses(inp,
                        m, cfg, reuse_weights = False)
            else:
                pred_pos = hidden_loop_with_bypasses(inp,
                        m, cfg, reuse_weights = True)
            pred.append(pred_pos)
        pred = tf.stack(pred, axis=1)
        pred = tf.expand_dims(pred, axis=2)
        retval = {'pred' : pred}
        retval.update(base_net.inputs)
        return retval, m.params


def rnn_more_data(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, obj_pic_dims = None, scale_down_height = None, scale_down_width = None, add_depth_gaussian = False, include_pose = False, **kwargs):
        batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
        long_len = inputs['object_data'].get_shape().as_list()[1]
        base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file, scale_down_height = scale_down_height, scale_down_width = scale_down_width, add_depth_gaussian = add_depth_gaussian)
        inputs = base_net.inputs

        size_1_attributes = ['normals', 'normals2', 'images']
        size_2_attributes = ['object_data_seen', 'actions_seen']
        size_1_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
        size_2_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_2_attributes], axis = 3) for t in range(time_seen)]
        m = ConvNetwithBypasses(**kwargs)

        encoded_input = []
        reuse_weights = False
        for t in range(time_seen):
                size_1_encoding_before_concat = feedforward_conv_loop(size_1_input_per_time[t], m, cfg, desc = 'size_1_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
                size_2_encoding_before_concat = feedforward_conv_loop(size_2_input_per_time[t], m, cfg, desc = 'size_2_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
                assert size_1_encoding_before_concat[-1].get_shape().as_list()[:-1] == size_2_encoding_before_concat[-1].get_shape().as_list()[:-1], (size_1_encoding_before_concat[-1].get_shape().as_list()[:-1], size_2_encoding_before_concat[-1].get_shape().as_list()[:-1])
                concat_inputs = tf.concat([size_1_encoding_before_concat[-1], size_2_encoding_before_concat[-1]], axis = 3)
                encoded_input.append(feedforward_conv_loop(concat_inputs, m, cfg, desc = 'encode', bypass_nodes = size_1_encoding_before_concat, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1])
                reuse_weights = True

        #flatten and concat
        flattened_input = [tf.reshape(enc_in, [batch_size, -1]) for enc_in in encoded_input]
        flattened_input.append(tf.reshape(inputs['object_data_seen_1d'], [batch_size, -1]))
        flattened_input.append(tf.reshape(inputs['depth_seen'], [batch_size, -1]))

        assert len(flattened_input[0].get_shape().as_list()) == 2
        concat_input = tf.concat(flattened_input, axis = 1)

        act = inputs['actions_no_pos']
        print('\033[91mHELLO!!!\033[0m')
        concat_input = tf.tile(tf.expand_dims(concat_input, 1),
                                [1, act.get_shape().as_list()[1], 1])
        rnn_input = tf.concat([concat_input, act], 2)
        pred, _ = rnn_loop(rnn_input, cfg)

        pred = tf.unstack(pred, axis=1)
        for i, _ in enumerate(pred):
            if i == 0:
                pred[i] = hidden_loop_with_bypasses(pred[i], 
                        m, cfg, reuse_weights = False)
            else:
                pred[i] = hidden_loop_with_bypasses(pred[i],
                        m, cfg, reuse_weights = True)
        pred = tf.stack(pred, axis=1)
        size = base_net.inputs['object_data_future'].get_shape().as_list()[1]
        begin = pred.get_shape().as_list()[1] - size
        pred = tf.slice(pred, [0,begin,0], [-1,size,-1])
        pred = tf.expand_dims(pred, axis=2)
        #pred = hidden_loop_with_bypasses(pred, m, cfg, reuse_weights = False)
        #pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
        #print(base_net.inputs['object_data_future'].get_shape().as_list())
        #if not include_pose:
        #        pred_shape[3] = 2
        #pred = tf.reshape(pred, pred_shape)

        

        retval = {'pred' : pred}
        retval.update(base_net.inputs)
        return retval, m.params



def include_more_data(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, obj_pic_dims = None, scale_down_height = None, scale_down_width = None, add_depth_gaussian = False, include_pose = False, **kwargs):
	batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
	long_len = inputs['object_data'].get_shape().as_list()[1]
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file, scale_down_height = scale_down_height, scale_down_width = scale_down_width, add_depth_gaussian = add_depth_gaussian)

	inputs = base_net.inputs

	size_1_attributes = ['normals', 'normals2', 'images']
	size_2_attributes = ['object_data_seen', 'actions_seen']
	size_1_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
	size_2_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_2_attributes], axis = 3) for t in range(time_seen)]
	m = ConvNetwithBypasses(**kwargs)

	encoded_input = []
	reuse_weights = False
	for t in range(time_seen):
		size_1_encoding_before_concat = feedforward_conv_loop(size_1_input_per_time[t], m, cfg, desc = 'size_1_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
		size_2_encoding_before_concat = feedforward_conv_loop(size_2_input_per_time[t], m, cfg, desc = 'size_2_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
		assert size_1_encoding_before_concat[-1].get_shape().as_list()[:-1] == size_2_encoding_before_concat[-1].get_shape().as_list()[:-1], (size_1_encoding_before_concat[-1].get_shape().as_list()[:-1], size_2_encoding_before_concat[-1].get_shape().as_list()[:-1])
		concat_inputs = tf.concat([size_1_encoding_before_concat[-1], size_2_encoding_before_concat[-1]], axis = 3)
		encoded_input.append(feedforward_conv_loop(concat_inputs, m, cfg, desc = 'encode', bypass_nodes = size_1_encoding_before_concat, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1])
		reuse_weights = True

	#flatten and concat
	flattened_input = [tf.reshape(enc_in, [batch_size, -1]) for enc_in in encoded_input]
	flattened_input.append(tf.reshape(inputs['object_data_seen_1d'], [batch_size, -1]))
	flattened_input.append(tf.reshape(inputs['actions_no_pos'], [batch_size, -1]))
	flattened_input.append(tf.reshape(inputs['depth_seen'], [batch_size, -1]))

	assert len(flattened_input[0].get_shape().as_list()) == 2
	concat_input = tf.concat(flattened_input, axis = 1)

	pred = hidden_loop_with_bypasses(concat_input, m, cfg, reuse_weights = False)
	pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
	if not include_pose:
		pred_shape[3] = 2
	pred = tf.reshape(pred, pred_shape)
	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params


def shared_weight_downscaled_nonimage(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, obj_pic_dims = None, scale_down_height = None, scale_down_width = None, **kwargs):
	batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
	long_len = inputs['object_data'].get_shape().as_list()[1]
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file, scale_down_height = scale_down_height, scale_down_width = scale_down_width)

	inputs = base_net.inputs

	size_1_attributes = ['normals', 'normals2']
	size_2_attributes = ['object_data_seen', 'actions_seen']
	size_1_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
	size_2_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_2_attributes], axis = 3) for t in range(time_seen)]
	m = ConvNetwithBypasses(**kwargs)

	encoded_input = []
	reuse_weights = False
	for t in range(time_seen):
		size_1_encoding_before_concat = feedforward_conv_loop(size_1_input_per_time[t], m, cfg, desc = 'size_1_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
		size_2_encoding_before_concat = feedforward_conv_loop(size_2_input_per_time[t], m, cfg, desc = 'size_2_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)
		assert size_1_encoding_before_concat[-1].get_shape().as_list()[:-1] == size_2_encoding_before_concat[-1].get_shape().as_list()[:-1], (size_1_encoding_before_concat[-1].get_shape().as_list()[:-1], size_2_encoding_before_concat[-1].get_shape().as_list()[:-1])
		concat_inputs = tf.concat([size_1_encoding_before_concat[-1], size_2_encoding_before_concat[-1]], axis = 3)
		encoded_input.append(feedforward_conv_loop(concat_inputs, m, cfg, desc = 'encode', bypass_nodes = size_1_encoding_before_concat, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1])
		reuse_weights = True

	#flatten and concat
	flattened_input = [tf.reshape(enc_in, [batch_size, -1]) for enc_in in encoded_input]
	flattened_input.append(tf.reshape(inputs['object_data_seen_1d'], [batch_size, -1]))

	if 'rnn' not in cfg:
            flattened_input.append(tf.reshape(
		inputs['actions_no_pos'], [batch_size, -1]))

	assert len(flattened_input[0].get_shape().as_list()) == 2
	concat_input = tf.concat(flattened_input, axis = 1)

        #recurrence
        if 'rnn' in cfg:
            act = inputs['actions_no_pos']
            concat_input = tf.tile(tf.expand_dims(concat_input, 1),
                    [1, act.get_shape().as_list()[1], 1])
            rnn_input = tf.concat([concat_input, act], 2)
            pred, _ = rnn_loop(rnn_input, cfg)
            pred = tf.expand_dims(pred, axis=2)
            if 'hidden' in cfg:
                pred = tf.reshape(pred, [batch_size, -1])
        else:
            pred = concat_input

        #hidden layers
        if 'hidden' in cfg:
            pred = hidden_loop_with_bypasses(pred, m, cfg, reuse_weights = False)  
            pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
	    pred_shape[3] = 2
	    pred = tf.reshape(pred, pred_shape)

	retval = {'pred' : pred}
	retval.update(base_net.inputs)
	return retval, m.params


def rnn_loop(inputs, cfg):
    for i in range(1, cfg['rnn_depth'] + 1):
        rnn_cfg = cfg['rnn'][i]
        if rnn_cfg['cell_type'] is 'gru':
            rnn_cell = tf.contrib.rnn.GRUCell(rnn_cfg['hidden_units'])
            initial_state = None
        elif rnn_cfg['cell_type'] is 'lstm':
            rnn_cell = tf.contrib.rnn.LSTMCell(rnn_cfg['hidden_units'])
            initial_state = (None, None)
        if 'dropout' in rnn_cfg:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell,
                    input_keep_prob = rnn_cfg['dropout']['input_keep_prob'],
                    output_keep_prob = rnn_cfg['dropout']['output_keep_prob'])
                    #state_keep_prob = rnn_cfg['dropout']['state_keep_prob'])
        inputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs,
                initial_state=initial_state,
                dtype = inputs.dtype)
        if 'output_format' in rnn_cfg:
            if rnn_cfg['output_format'] is 'last':
                outputs = tf.unstack(inputs, axis=1)
                return [outputs[-1], state]
            else:
                 raise ValueError('RNN: Unknown output format')
        else:
            outputs = inputs
            return [outputs, state]


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
		diffed_data_list.append(future_data[:, t:t+1] - last_seen_data)
		last_seen_data = future_data[:, t:t+1]
	print(diffed_data_list)
	return tf.concat(diffed_data_list, axis = 1)

def other_diffs(last_seen_data, future_data):
	return tf.concat([future_data[:, t : t + 1] - last_seen_data for t in range(future_data.get_shape().as_list()[1])], axis = 1)

def l2_diff_loss(outputs):
	pred = outputs['pred']
	future_dat = outputs['object_data_future']
	seen_dat = outputs['object_data_seen_1d']
	last_seen_dat = seen_dat[:, -1:]
	tv = compute_diffs(last_seen_dat, future_dat)
	tv = tf.reshape(tv, [tv.get_shape().as_list()[0], -1])
	n_entries = tv.get_shape().as_list()[1] * tv.get_shape().as_list()[0]
	return 100. * tf.nn.l2_loss(pred - tv) / n_entries # now with a multiplier because i'm tired of staring at tiny numbers


def l2_diff_loss_just_positions(outputs):
	pred = outputs['pred']
	future_dat = outputs['object_data_future']
	seen_dat = outputs['object_data_seen_1d']
	last_seen_dat = seen_dat[:, -1:]
	tv = compute_diffs(last_seen_dat, future_dat)
	print('tv shape!')
	print(tv.get_shape().as_list())
	tv = tv[:, :, :, 4:]
	tv = tf.reshape(tv, [tv.get_shape().as_list()[0], -1])
	n_entries = tv.get_shape().as_list()[1] * tv.get_shape().as_list()[0]
	return 100. * tf.nn.l2_loss(pred - tv) / n_entries # now with a multiplier because i'm tired of staring at tiny numbers

def alternate_diff_loss(outputs):
	pred = outputs['pred']
	future_dat = outputs['object_data_future']
	seen_dat = outputs['object_data_seen_1d']
	last_seen_dat = seen_dat[:, -1:]
	tv = other_diffs(last_seen_dat, future_dat)
	tv = tv[:, :, :, 4:]
	tv = tf.reshape(tv, [tv.get_shape().as_list()[0], -1])
	n_entries = tv.get_shape().as_list()[1] * tv.get_shape().as_list()[0]
	return 100. * tf.nn.l2_loss(pred - tv) / n_entries # now with a multiplier because i'm tired of staring at tiny numbers

def diff_loss_with_mask(outputs):
	master_filter = outputs['master_filter']
	pred = outputs['pred']
	future_dat = outputs['object_data_future']
	seen_dat = outputs['object_data_seen_1d']
	time_seen = seen_dat.get_shape().as_list()[1]
	last_seen_dat = seen_dat[:, -1:]
	tv = compute_diffs(last_seen_dat, future_dat)
	tv = tv[:, :, :, 4:]
	mask = tf.cast(tf.cumprod(master_filter[:, time_seen:], axis = 1), tf.float32)
	mask = tf.expand_dims(mask, axis = 2)
	mask = tf.expand_dims(mask, axis = 2)
	mask = tf.tile(mask, [1, 1, 1, 2])
	n_entries = np.prod(tv.get_shape().as_list())
	return 100. * tf.nn.l2_loss(mask * (pred - tv)) / n_entries

def diff_mask_loss_with_poses(outputs):
	master_filter = outputs['master_filter']
	pred = outputs['pred']
	future_dat = outputs['object_data_future']
	seen_dat = outputs['object_data_seen_1d']
	time_seen = seen_dat.get_shape().as_list()[1]
	last_seen_dat = seen_dat[:, -1:]
	tv = compute_diffs(last_seen_dat, future_dat)
	mask = tf.cast(tf.cumprod(master_filter[:, time_seen:], axis = 1), tf.float32)
	mask = tf.expand_dims(mask, axis = 2)
	mask = tf.expand_dims(mask, axis = 2)
	mask = tf.tile(mask, [1, 1, 1, 6])
	n_entries = np.prod(tv.get_shape().as_list())
	return 100. * tf.nn.l2_loss(mask * (pred - tv)) / n_entries

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

cfg_2 = {
	'encode_depth' : 7,
	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 32}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 32}},
		3 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 32}},
		4 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 16}},
		5 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 16}},
		6 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 16}},
		7 : {'conv' : {'filter_size' : 7, 'stride' : 1, 'num_filters' : 16}}
	},

	'hidden_depth' : 2,
	'hidden' : {
		1 : {'num_features' : 70},
		2 : {'num_features' : 60, 'activation' : 'identity'} #2 points * 5 timesteps * (2 + 4) dimension
	}
}

cfg_share_to_mlp = {
	'encode_depth' : 7,
	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 8}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 8}},
		3 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 8}},
		4 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 5}},
		5 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 5}},
		6 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 5}},
		7 : {'conv' : {'filter_size' : 7, 'stride' : 1, 'num_filters' : 5}}
	},

	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 160},
		2 : {'num_features' : 160},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}	
}

#so small size shapes should be int(160 / 4) = 40 by int(375 / 4) = 94
cfg_simple_different_sizes = {
	'size_1_before_concat_depth' : 3,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		3 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 6}, 'bypass' : [0]}
	},

	'size_2_before_concat_depth' : 0,

	'encode_depth' : 5,

	'encode' : {
		1 : {'conv' : {'filter_size' : 11, 'stride' : 2, 'num_filters' : 32}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}}, # size 256 image, this leads to 16 * 16 * 256 = 65,536 neurons. Sad!
	},

	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 250},
		2 : {'num_features' : 250},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}	
}

cfg_fewer_channels_different_sizes = {
	'size_1_before_concat_depth' : 3,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		3 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 6}, 'bypass' : [0]}
	},

	'size_2_before_concat_depth' : 0,

	'encode' : {
		1 : {'conv' : {'filter_size' : 11, 'stride' : 1, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 2}},
	},

	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 250},
		2 : {'num_features' : 250},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}	

}



cfg_onetwoone_short = {

	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'coord_to_conv_depth' : 1,
	'coord_to_conv' : {
		0 : {'out_shape' : [40, 94, 6], 'activation' : 'relu'},
		1 : {'conv'  : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 4}}
	},

	'encode_depth' : 2,

	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}, 'bypass' : 0},
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}

}


cfg_one_to_two_to_one = {

	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'coord_to_conv_depth' : 1,
	'coord_to_conv' : {
		0 : {'out_shape' : [40, 94, 4], 'activation' : 'relu'},
		1 : {'conv'  : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 4}}
	},


	'encode_depth' : 4 + 4 + 4 + 4 + 1,

	'encode' : {
		1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}},
		2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}, 'bypass' : -3},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}, 'bypass' : -3},
		6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}, 'bypass' : -3},
		8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		9 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 8}, 'bypass' : -3},
		10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		12 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		13 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 4}, 'bypass' : -3},
		14 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		15 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		16 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		17 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3}
	},





	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 250},
		2 : {'num_features' : 250},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}	


}


cfg_more_bypasses = {
	'size_1_before_concat_depth' : 3,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		3 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 6}, 'bypass' : [0]}
	},

	'size_2_before_concat_depth' : 0,

	'encode_depth' : 5,

	'encode' : {
		1 : {'conv' : {'filter_size' : 11, 'stride' : 1, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}, 'bypass' : [0]},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}, 'bypass' : [0]},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}, 'bypass' : [0]},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : [0]},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 3}, 'bypass' : [0]}, # size 256 image, this leads to 16 * 16 * 256 = 65,536 neurons. Sad!
	},

	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 250},
		2 : {'num_features' : 250},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}
}

cfg_more_bypasses_smaller_end = {
	'size_1_before_concat_depth' : 3,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}},
		3 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 6}, 'bypass' : [0]}
	},

	'size_2_before_concat_depth' : 0,

	'encode' : {
		1 : {'conv' : {'filter_size' : 11, 'stride' : 2, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}, 'bypass' : [0]},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}, 'bypass' : [0]},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}, 'bypass' : [0]},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : [0]},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : [0]},
	},

	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 250},
		2 : {'num_features' : 250},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}	
}


cfg_alexy_share_to_mlp = {
	'encode_depth' : 5,
	'encode' : {
		1 : {'conv' : {'filter_size' : 11, 'stride' : 4, 'num_filters' : 32}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 16}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}}, # size 256 image, this leads to 16 * 16 * 256 = 65,536 neurons. Sad!
	},

	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 250},
		2 : {'num_features' : 250},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}	



}

cfg_mlp = {
	'hidden_depth' : 2,
	'hidden' : {
		1 : {'num_features' : 70},
		2 : {'num_features' : 60, 'activation' : 'identity'}
	}
}

cfg_mlp_med = {
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 160},
		2 : {'num_features' : 160},
		3 : {'num_features' : 60, 'activation' : 'identity'}
	}
}

cfg_mlp_wide = {
	'hidden_depth' : 3,
	'hidden' : {
		1 : {'num_features' : 300},
		2 : {'num_features' : 300},
		3 : {'num_features' : 60, 'activation' : 'identity'}
	}
}

cfg_mlp_interesting_nonlinearities = {
	'hidden_depth' : 3,
	'hidden' : {
		1 : {'num_features' : 100, 'activation' : ['square', 'identity', 'relu']},
		2 : {'num_features' : 100, 'activation' : ['square', 'identity', 'relu']},
		3 : {'num_features' : 60, 'activation' : 'identity'}
	}
}

cfg_mlp_med_just_positions = {
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 160},
		2 : {'num_features' : 160},
		3 : {'num_features' : 20, 'activation' : 'identity'}
	}	
}

cfg_mlp_interesting_nonlinearities_just_positions = {
	'hidden_depth' : 3,
	'hidden' : {
		1 : {'num_features' : 100, 'activation' : ['square', 'identity', 'relu']},
		2 : {'num_features' : 100, 'activation' : ['square', 'identity', 'relu']},
		3 : {'num_features' : 20, 'activation' : 'identity'}
	}
}

cfg_mlp_wide_just_positions = {
	'hidden_depth' : 3,
	'hidden' : {
		1 : {'num_features' : 300},
		2 : {'num_features' : 300},
		3 : {'num_features' : 20, 'activation' : 'identity'}
	}
}

cfg_mlp_med_just_positions_one_obj = {
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 160},
		2 : {'num_features' : 160},
		3 : {'num_features' : 10, 'activation' : 'identity'}
	}	
}

cfg_mlp_med_more_timesteps = {
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 160},
		2 : {'num_features' : 160},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}	
}


cfg_mlp_wider = {
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}



}

cfg_resnet18 = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 4 + 4 + 4 + 4 + 1,

	'encode' : {
		1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}, 'bypass' : -3},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16}, 'bypass' : -3},
		6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}, 'bypass' : -3},
		8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		9 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 8}, 'bypass' : -3},
		10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		12 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		13 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 4}, 'bypass' : -3},
		14 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		15 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		16 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		17 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3}
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 250},
		2 : {'num_features' : 250},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}

}

cfg_resnet_wide = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 4 + 4 + 4 + 4 + 1,

	'encode' : {
		1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}, 'bypass' : -3},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16}, 'bypass' : -3},
		6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}, 'bypass' : -3},
		8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		9 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 8}, 'bypass' : -3},
		10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		12 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		13 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 4}, 'bypass' : -3},
		14 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		15 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		16 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		17 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3}
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
            1: {'num_features' : 1000},
            2 : {'num_features' : 1000},
            3 : {'num_features' : 40, 'activation' : 'identity'}
        },
        'rnn_depth': 1,
        'rnn': {
            1: {'cell_type': 'gru',
                'hidden_units': 1000,
                'dropout': {
                    'input_keep_prob': 1.0,
                    'output_keep_prob': 1.0,
                    #'state_keep_prob': 1.0,
                    }
            },
        }
}

cfg_resnet_interesting_nonlinearities = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 4 + 4 + 4 + 4 + 1,

	'encode' : {
		1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}, 'bypass' : -3},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16}, 'bypass' : -3},
		6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}, 'bypass' : -3},
		8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}},
		9 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 8}, 'bypass' : -3},
		10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		12 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		13 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 4}, 'bypass' : -3},
		14 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		15 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3},
		16 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 4}},
		17 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 4}, 'bypass' : -3}
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1 : {'num_features' : 500, 'activation' : ['square', 'identity', 'relu']},
		2 : {'num_features' : 500, 'activation' : ['square', 'identity', 'relu']},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}


}

cfg_resnet_more_channels = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6 * 4}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 4 + 4 + 4 + 4 + 1,

	'encode' : {
		1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}},
		2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}, 'bypass' : -3},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 34 * 2}, 'bypass' : -3},
		6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 2}},
		7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 2}, 'bypass' : -3},
		8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 2}},
		9 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 34 * 4}, 'bypass' : -3},
		10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 4}},
		11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 4}, 'bypass' : -3},
		12 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 4}},
		13 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 34 * 8}, 'bypass' : -3},
		14 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 8}},
		15 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 8}, 'bypass' : -3},
		16 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 8}},
		17 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 34 * 8}, 'bypass' : -3}
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1 : {'num_features' : 500, 'activation' : ['square', 'identity', 'relu']},
		2 : {'num_features' : 500, 'activation' : ['square', 'identity', 'relu']},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}


}

cfg_short_conv_one_step = {
        'size_1_before_concat_depth' : 1,

        'size_1_before_concat' : {
                1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },


        'size_2_before_concat_depth' : 0,

        'encode_depth' : 2,

        'encode' : {
                1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}},
                2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}, 'bypass' : 0},
        },
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
        'hidden_depth' : 10,
        'hidden' : {
                1 : {'num_features' : 100},
                2 : {'num_features' : 100},
                3 : {'num_features' : 100},
                4 : {'num_features' : 100},
                5 : {'num_features' : 100},
                6 : {'num_features' : 100},
                7 : {'num_features' : 100},
                8 : {'num_features' : 100},
                9 : {'num_features' : 100},
                10 : {'num_features' : 2, 'activation' : 'identity'}
        },
        'rnn_depth': 0,
        'rnn': {
            1: {'cell_type': 'gru',
                'hidden_units': 1000,
                'dropout': {
                    'input_keep_prob': 1.0,
                    'output_keep_prob': 1.0,
                    #'state_keep_prob': 1.0,
                    }
            },
        }
}

cfg_short_conv_rnn = {
        'size_1_before_concat_depth' : 1,

        'size_1_before_concat' : {
                1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },


        'size_2_before_concat_depth' : 0,

        'encode_depth' : 2,

        'encode' : {
                1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}},
                2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}, 'bypass' : 0},
        },
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
        'hidden_depth' : 1,
        'hidden' : {
                 1 : {'num_features' : 2, 'activation': 'identity'},
        #        2 : {'num_features' : 1000},
        #        3 : {'num_features' : 40, 'activation' : 'identity'}
        },
        'rnn_depth': 1,
        'rnn': {
            1: {'cell_type': 'gru',
                'hidden_units': 1000,
                'dropout': {
                    'input_keep_prob': 1.0,
                    'output_keep_prob': 1.0,
                    #'state_keep_prob': 1.0,
                    }
            },
        }
}

cfg_short_conv = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 2,

	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}, 'bypass' : 0},
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}
}

cfg_less_short_conv = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 3,

	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 1, 'num_filters' : 34}},
		3 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}, 'bypass' : 0},
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}

}

cfg_even_shorter_conv = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 1,

	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 4, 'num_filters' : 34}}
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}
}


cfg_short_conv_w_poses = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 2,

	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 34}, 'bypass' : 0},
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1200},
		2 : {'num_features' : 1200},
		3 : {'num_features' : 120, 'activation' : 'identity'}
	}

}




cfg_short_conv_more_info = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 72}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 2,

	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 102}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 102}, 'bypass' : 0},
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}

}

cfg_short_conv_more_info_fewer_channels = {
	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 36}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'size_2_before_concat_depth' : 0,

	'encode_depth' : 2,

	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 51}},
		2 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 51}, 'bypass' : 0},
	},
#down to 5 x 12 x 4
#this end stuff is where we should maybe join time steps
	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}

}


cfg_121_channels = {

	'size_1_before_concat_depth' : 1,

	'size_1_before_concat' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 6}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
	},


	'coord_to_conv_depth' : 1,
	'coord_to_conv' : {
		0 : {'out_shape' : [40, 94, 4], 'activation' : 'relu'},
		1 : {'conv'  : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 4}}
	},


	'encode_depth' : 4 + 4 + 4 + 4 + 1,

	'encode' : {
		1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}},
		2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}, 'bypass' : -3},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 34 * 2}, 'bypass' : -3},
		6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 2}},
		7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 2}, 'bypass' : -3},
		8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 2}},
		9 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 34 * 4}, 'bypass' : -3},
		10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 4}},
		11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 4}, 'bypass' : -3},
		12 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 4}},
		13 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 34 * 8}, 'bypass' : -3},
		14 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 8}},
		15 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 8}, 'bypass' : -3},
		16 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 34 * 8}},
		17 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 34 * 8}, 'bypass' : -3}
	},





	'hidden_depth' : 3,
	'hidden' : {
		1: {'num_features' : 1000},
		2 : {'num_features' : 1000},
		3 : {'num_features' : 40, 'activation' : 'identity'}
	}

}
