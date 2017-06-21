'''
Policy and intrinsic reward models.
'''

import numpy as np
import tensorflow as tf
from curiosity.models.model_building_blocks import ConvNetwithBypasses

class UniformActionSampler:
	def __init__(self, cfg):
		self.action_dim = cfg['uncertainty_model']['action_dim']
		self.num_actions = cfg['uncertainty_model']['n_action_samples']
		self.rng = np.random.RandomState(cfg['seed'])

	def sample_actions(self):
		return self.rng.uniform(-1., 1., (self.num_actions, self.action_dim))

def postprocess_depths(depths):
	'''
		Assumes depths is of shape [batch_size, time_number, height, width, 3]
	'''
	depths = tf.cast(depths, tf.float32)
	depths = (depths[:,:,:,:,0:1] * 256 + depths[:,:,:,:,1:2] + \
	        depths[:,:,:,:,2:3] / 256.0) / 1000.0 
	depths /= 17.32 # normalization
	return depths


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])



def categorical_sample(logits, d, one_hot = True):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    if not one_hot:
    	return value
    return tf.one_hot(value, d)

def deconv_loop(input_node, m, cfg, desc = 'deconv', bypass_nodes = None,
        reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print = True, return_bypass=False, sub_bypass = None):
    m.output = input_node
    deconv_nodes = [input_node]
    # deconvolving
    deconv_depth = cfg[desc + '_depth']
    print('deconv depth: %d' % deconv_depth)
    cfs0 = None

    if bypass_nodes is None:
        bypass_nodes = [m.output]

    for i in range(1, deconv_depth + 1):
        with tf.variable_scope(desc + str(i)) as scope:
            if reuse_weights:
                scope.reuse_variables()

            bypass = cfg[desc][i].get('bypass')
            if bypass:
                if type(bypass) == list:
                    bypass_node = [bypass_nodes[bp] for bp in bypass]
                elif type(bypass) == dict:
                    if sub_bypass is None:
                       raise ValueError('Bypass \
                               is dict but no sub_bypass specified')
                    for k in bypass:
                        if int(k) == sub_bypass:
                            if type(bypass[k]) == list:
                                bypass_node = [bypass_nodes[bp] \
                                        for bp in bypass[k]]
                            else:
                                bypass_node = bypass_nodes[bypass[k]]
                else:
                    bypass_node = bypass_nodes[bypass]
                m.add_bypass(bypass_node)

            bn = cfg[desc][i]['deconv'].get('batch_normalize')
            if bn:
                norm_it = bn
            else:
                norm_it = batch_normalize

            with tf.contrib.framework.arg_scope([m.deconv], 
                    init='xavier', stddev=.01, bias=0, batch_normalize = norm_it):
                cfs = cfg[desc][i]['deconv']['filter_size']
                cfs0 = cfs
                nf = cfg[desc][i]['deconv']['num_filters']
                cs = cfg[desc][i]['deconv']['stride']
                if 'output_shape' in cfg[desc][i]['deconv']:
                    out_shape = cfg[desc][i]['deconv']['output_shape']
                else:
                    out_shape = None
                if do_print:
                    print('deconv in: ', m.output)
                if no_nonlinearity_end and i == deconv_depth:
                    m.deconv(nf, cfs, cs, activation = None, 
                            fixed_output_shape=out_shape)
                else:
                    my_activation = cfg[desc][i].get('nonlinearity')
                    if my_activation is None:
                        my_activation = 'relu'
                    m.deconv(nf, cfs, cs, activation = my_activation, 
                            fixed_output_shape=out_shape)
                    if do_print:
                        print('deconv out:', m.output)
                    #TODO add print function
                    pool = cfg[desc][i].get('pool')
                    if pool:
                        pfs = pool['size']
                        ps = pool['stride']
                        m.pool(pfs, ps)
                    deconv_nodes.append(m.output)
                    bypass_nodes.append(m.output)
    if return_bypass:
        return [deconv_nodes, bypass_nodes]
    return deconv_nodes



def feedforward_conv_loop(input_node, m, cfg, desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print=True, return_bypass=False, sub_bypass = None):
        m.output = input_node
        encode_nodes = [input_node]
        #encoding
        encode_depth = cfg[desc + '_depth']
        print('conv depth: %d' % encode_depth)
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
                                elif type(bypass) == dict:
                                    if sub_bypass is None:
                                        raise ValueError('Bypass \
                                                is dict but no sub_bypass specified')
                                    for k in bypass:
                                        if int(k) == sub_bypass:
                                            if type(bypass[k]) == list:
                                                bypass_node = [bypass_nodes[bp] \
                                                        for bp in bypass[k]]
                                            else:
                                                bypass_node = bypass_nodes[bypass[k]]
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
                            if do_print:
                                print('conv in', m.output)
                            if no_nonlinearity_end and i == encode_depth:
                                m.conv(nf, cfs, cs, activation = None)
                            else:
                                my_activation = cfg[desc][i].get('nonlinearity')
                                if my_activation is None:
                                        my_activation = 'relu'
                                else:
                                        print('NONLIN: ' + my_activation)
                                m.conv(nf, cfs, cs, activation = my_activation)
                            if do_print:
                                print('conv out', m.output)
        #TODO add print function
                        pool = cfg[desc][i].get('pool')
                        if pool:
                            pfs = pool['size']
                            ps = pool['stride']
                            m.pool(pfs, ps)
                        encode_nodes.append(m.output)
                        bypass_nodes.append(m.output)
        if return_bypass:
            return [encode_nodes, bypass_nodes]
        return encode_nodes


def hidden_loop_with_bypasses(input_node, m, cfg, nodes_for_bypass = [], stddev = .01, reuse_weights = False, activation = 'relu', train = False):
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
                        if train:
                                my_dropout = cfg['hidden'][i].get('dropout')
                        else:
                                my_dropout = None
                        print(nf)
                        print(my_activation)
                        print(stddev)
                        print(my_dropout)
                        m.fc(nf, init = 'xavier', activation = my_activation, bias = .01, stddev = stddev, dropout = my_dropout)
                        nodes_for_bypass.append(m.output)
                        print(m.output)
        return m.output


def flatten_append_unflatten(start_state, action, cfg, m):
	x = flatten(start_state)
	joined = tf.concat(1, [x, action])
	x = hidden_loop_with_bypasses(joined, m, cfg['mlp'], reuse_weights = False, train = True)

	reshape_dims = cfg['reshape_dims']
	# assert np.prod(reshape_dims) == tf.shape(joined)[-1],  (np.prod(reshape_dims), tf.shape(joined)[-1])
	return tf.reshape(x, [-1] + reshape_dims)


class DepthFuturePredictionWorldModel():
	def __init__(self, cfg, action_state_join_model = flatten_append_unflatten):
		print('Warning! dropout train/test not currently being handled.')
		with tf.variable_scope('wm'):
			self.s_i = x = tf.placeholder(tf.float32, [1] + cfg['state_shape'])
			self.s_f = s_f = tf.placeholder(tf.float32, [1] + cfg['state_shape'])
			self.action = tf.placeholder(tf.float32, [1, cfg['action_dim']])
			bs = tf.to_float(tf.shape(self.s_i)[0])
			#convert from 3-channel encoding
			self.processed_input = x = postprocess_depths(x)

			s_f = postprocess_depths(s_f)
			#flatten time dim
			x = tf.concat(3, [x[:, i] for i in range(cfg['state_shape'][0])])
			#encode
			m = ConvNetwithBypasses()
			all_encoding_layers = feedforward_conv_loop(x, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False)
			x = all_encoding_layers[-1]

			joined = action_state_join_model(x, self.action, cfg['action_join'], m)

			decoding = deconv_loop(
	                            joined, m, cfg['deconv'], desc='deconv',
	                            bypass_nodes = all_encoding_layers, reuse_weights = False,
	                            batch_normalize = False,
	                            do_print = True)
			self.pred = decoding[-1]
			self.tv = s_f[:, -1]
			self.loss = tf.nn.l2_loss(self.tv - self.pred) / (bs * np.prod(cfg['state_shape']))



sample_depth_future_cfg = {
	'state_shape' : [2, 64, 64, 3],
	'action_dim' : 8,
	'action_join' : {
		'reshape_dims' : [8, 8, 5],

		'mlp' : {
			'hidden_depth' : 2,
			'hidden' : {
				1 : {'num_features' : 320, 'dropout' : .75},
				2 : {'num_features' : 320, 'activation' : 'identity'}
			}
		}
	},

	'encode' : {
		'encode_depth' : 3,
		'encode' : {
			1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}},
			2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}},
			3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 5}},
		}
	},

	'deconv' : {
		'deconv_depth' : 3,

		'deconv' : {
			1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}, 'bypass' : 0},
			2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}, 'bypass' : 0},
			3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 1}, 'bypass' : 0}
		}


	}
}


a_bigger_depth_future_config = {
	'state_shape' : [2, 64, 64, 3],
	'action_dim' : 8,

	'action_join' : {
		'reshape_dims' : [8, 8, 5],

		'mlp' : {
			'hidden_depth' : 3,
			'hidden' : {
				1 : {'num_features' : 320},
				2 : {'num_features' : 320},
				3 : {'num_features' : 320, 'activation' : 'identity'}
			}
		}
	},

	'encode' : {
		'encode_depth' : 5,
		'encode' : {
			1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 20}},
			2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 20}},
			3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 20}},
			4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}},
			5 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 5}},
		}
	},

	'deconv' : {
		'deconv_depth' : 5,

		'deconv' : {
			1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 20}, 'bypass' : 0},
			2 : {'deconv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 20}, 'bypass' : 0},
			3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 20}, 'bypass' : 0},
			4 : {'deconv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}, 'bypass' : 0},
			5 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 1}, 'bypass' : 0}
		}


	}


}






class WorldModel(object):
    def __init__(self, ob_space, ac_space):
        self.s_i = s_i = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.s_f = s_f = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action_one_hot = tf.placeholder(tf.float32, [None] + [ac_space])
        self.encode_var_list = []

        for i in range(4):
            s_i, w, b = conv2d(s_i, 32, "encode{}".format(i + 1), [3, 3], [2, 2])
            s_i = tf.nn.elu(s_i)
            self.encode_var_list = self.encode_var_list + [w, b]
            s_f, _, _ = conv2d(s_f, 32, "encode{}".format(i + 1), [3, 3], [2, 2], reuse_weights = True)
            s_f = tf.nn.elu(s_f)

        encoding_i = flatten(s_i)
        self.encoding_f = flatten(s_f)
        print(encoding_i)
        print(self.encoding_f)

        #predicting action
        act_input = tf.concat(1, [encoding_i, self.encoding_f])
        print(act_input)
        act_hidden, w_act_h, b_act_h = linear(act_input, 256, 'worldact_h', normalized_columns_initializer(0.01))
        act_hidden = tf.nn.elu(act_hidden)
        self.act_logits, w_act, b_act = linear(act_hidden, ac_space, 'worldact_out', normalized_columns_initializer(0.01))
        self.act_var_list = [w_act_h, b_act_h, w_act, b_act]

        self.act_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.action_one_hot, logits = self.act_logits))

        #forward model
        fut_input = tf.concat(1, [encoding_i, self.action_one_hot])
        fut_hidden, w_fut_h, b_fut_h = linear(fut_input, 256, 'worldfut_h', normalized_columns_initializer(0.01))
        fut_hidden = tf.nn.elu(fut_hidden)
        #not really sure if there's some good intuition for initializers here
        self.fut_pred, w_fut, b_fut = linear(fut_hidden, 288, 'worldfut_out', normalized_columns_initializer(0.01))
        self.fut_loss = tf.nn.l2_loss(self.encoding_f - self.fut_pred) / (100.)
        self.fut_var_list = [w_fut_h, b_fut_h, w_fut, b_fut]


class UncertaintyModel:
	def __init__(self, cfg):
		with tf.variable_scope('um'):
			self.s_i = x = tf.placeholder(tf.float32, [1] + cfg['state_shape'])
			self.action_sample = ac = tf.placeholder(tf.float32, [None, cfg['action_dim']])
			self.true_loss = tr_loss = tf.placeholder(tf.float32, [1])
			m = ConvNetwithBypasses()
			x = postprocess_depths(x)
			#concatenate temporal dimension into channels
			x = tf.concat(3, [x[:, i] for i in range(cfg['state_shape'][0])])
			#encode
			self.encoded = x = feedforward_conv_loop(x, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False)[-1]
			x = flatten(x)
			x = tf.cond(tf.shape(self.action_sample)[0] > 1, lambda : tf.tile(x, [cfg['n_action_samples'], 1]), lambda : x)
			# x = tf.tile(x, [cfg['n_action_samples'], 1])
			x = tf.concat(1, [x, ac])
			self.estimated_world_loss = x = hidden_loop_with_bypasses(x, m, cfg['mlp'], reuse_weights = False, train = True)
			x_tr = tf.transpose(x)
			self.sample = categorical_sample(x_tr, cfg['n_action_samples'], one_hot = False)
			self.uncertainty_loss = tf.nn.l2_loss(self.estimated_world_loss - self.true_loss)

	def act(self, sess, action_sample, state):
		chosen_idx = sess.run(self.sample, feed_dict = {self.s_i : state, self.action_sample : action_sample})[0]
		return action_sample[chosen_idx]

sample_cfg = {
	'uncertainty_model' : {
		'state_shape' : [2, 64, 64, 3],
		'action_dim' : 8,
		'n_action_samples' : 50,
		'encode' : {
			'encode_depth' : 3,
			'encode' : {
				1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}},
				2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}},
				3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 5}},
			}
		},
		'mlp' : {
			'hidden_depth' : 2,
			'hidden' : {1 : {'num_features' : 20, 'dropout' : .75},
						2 : {'num_features' : 1, 'activation' : 'identity'}
			}		
		}
	},

	'world_model' : sample_depth_future_cfg,

	'seed' : 0
}


class LSTMDiscretePolicy:
	def __init__(self, cfg):
		self.x = x = tf.placeholder(tf.float32, [None] + cfg['state_shape'])
		m = ConvNetwithBypasses(**kwargs)
		x = feedforward_conv_loop(x, m, cfg, desc = 'size_1_before_concat', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False, no_nonlinearity_end = False)[-1] 

		x = tf.expand_dims(flatten(x), [0])

		lstm_size = cfg['lstm_size']
		if use_tf100_api:
			lstm = rnn.BasicLSTMCell(lstm_size, state_is_tuple = True)
		else:
			lstm = rnn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple = True)
		self.state_size = lstm.state_size

		c_init = np.zeros((1, lstm.state_size.c), np.float32)
		h_init = np.zeros((1, lstm.state_size.h), np.float32)
		self.state_init = [c_init, h_init]
		c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
		h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
		self.state_in = [c_in, h_in]

		if use_tf100_api:
		    state_in = rnn.LSTMStateTuple(c_in, h_in)
		else:
		    state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
		lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
		    lstm, x, initial_state=state_in, sequence_length=step_size,
		    time_major=False)
		lstm_c, lstm_h = lstm_state
		self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

		x = tf.reshape(lstm_outputs, [-1, size])
		self.vf = hidden_loop_with_bypasses(x, m, cfg['value'], reuse_weights = False, train = True)
		self.logits = hidden_loop_with_bypasses(x, m, cfg['logits'], reuse_weights = False, train = True)
		self.sample = categorical_sample(self.logits, ac_space)[0, :]
		self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)



	def get_initial_features(self):
	    return self.state_init

	def act(self, ob, c, h):
	    sess = tf.get_default_session()
	    return sess.run([self.sample, self.vf] + self.state_out,
	                    {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

	def value(self, ob, c, h):
	    sess = tf.get_default_session()
	    return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
