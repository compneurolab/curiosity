'''
Takes some from twoobject_to_oneobject.py, but that thing was getting too long.
Jerk prediction models.
'''

import explicit_future_prediction_base as fp_base
import tensorflow as tf
from curiosity.models.model_building_blocks import ConvNetwithBypasses
import numpy as np
import cPickle

def deconv_loop(input_node, m, cfg, desc = 'deconv', bypass_nodes = None,
        reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print = True, return_bypass=False):
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

def feedforward_conv_loop(input_node, m, cfg, desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print=True, return_bypass=False):
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
                        m.fc(nf, init = 'xavier', activation = my_activation, bias = .01, stddev = stddev, dropout = my_dropout)
                        nodes_for_bypass.append(m.output)
                        print(m.output)
        return m.output


def basic_jerk_bench(inputs, cfg = None, num_classes = None, time_seen = None, normalization_method = None, stats_file = None, add_gaussians = True, image_height = None, image_width = None, **kwargs):
        base_net = fp_base.ShortLongFuturePredictionBase(inputs, store_jerk = True, normalization_method = normalization_method,
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
        pred = hidden_loop_with_bypasses(m.output, m, cfg, reuse_weights = False, train = kwargs['train'])
        pred_shape = pred.get_shape().as_list()
        if num_classes is not None:
                pred_shape.append(int(num_classes))
                pred_shape[1] = int(pred_shape[1] / num_classes)
                pred = tf.reshape(pred, pred_shape)
        retval = {'pred' : pred}
        retval.update(base_net.inputs)
        return retval, m.params

def map_jerk_model(inputs, cfg = None, time_seen = None, normalization_method = None, 
        stats_file = None, obj_pic_dims = None, scale_down_height = None, 
        scale_down_width = None, add_depth_gaussian = False, include_pose = False, 
        num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
    print('------NETWORK START-----')
    with tf.device('/gpu:%d' % gpu_id):
        # rescale inputs to be divisible by 8
        rinputs = {}
        for k in inputs:
            if k in ['depths', 'objects']:
                rinputs[k] = tf.pad(inputs[k], 
                        [[0,0], [0,0], [0,0], [3,3], [0,0]], "CONSTANT")
            else:
                rinputs[k] = inputs[k]
       # preprocess input data         
        batch_size, time_seen = rinputs['depths'].get_shape().as_list()[:2]
        time_seen -= 1
        long_len = rinputs['object_data'].get_shape().as_list()[1]
        base_net = fp_base.ShortLongFuturePredictionBase(
                rinputs, store_jerk = True, 
                normalization_method = normalization_method,
                time_seen = time_seen, stats_file = stats_file, 
                scale_down_height = scale_down_height, 
                scale_down_width = scale_down_width, 
                add_depth_gaussian = add_depth_gaussian)
        inputs = base_net.inputs

        # init network
        m = ConvNetwithBypasses(**kwargs)
        # encode per time step
        size_1_attributes = ['depths'] 
        size_1_input_per_time = [tf.concat([inputs[nm][:, t] \
                for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
        encoded_input = []
        bypass_nodes = []
        reuse_weights = False
        do_print = True
        for t in range(time_seen):
                enc, bypass_nodes = feedforward_conv_loop(
                        size_1_input_per_time[t], m, cfg, desc = 'encode', 
                        bypass_nodes = None, reuse_weights = reuse_weights, 
                        batch_normalize = False, no_nonlinearity_end = False, 
                        do_print=do_print, return_bypass = True)
                encoded_input.append(enc[-1])
                do_print = False
                reuse_weights = True
        # encode across time
        num_encode_together = cfg.get('encode_together_depth')
        if num_encode_together:
                print('Encoding across time')
                together_input = tf.concat(encoded_input, axis = 3)
                encoded_input, bypass_nodes = feedforward_conv_loop(
                        together_input, m, cfg, desc = 'encode_together', 
                        bypass_nodes = bypass_nodes, reuse_weights = False, 
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print = True, return_bypass = True)
                encoded_input = encoded_input[-1:]
        # decode
        num_deconv = cfg.get('deconv_depth')
        if num_deconv:
            encoded_input, bypass_nodes = deconv_loop(
                    encoded_input[-1], m, cfg, desc='deconv', 
                    bypass_nodes = bypass_nodes, reuse_weights = False, 
                    batch_normalize = False, no_nonlinearity_end = False,
                    do_print = True, return_bypass = True)
            encoded_input = encoded_input[-1:]
        # return prediction output
        pred = encoded_input[-1]
        retval = {'pred' : pred}
        retval.update(base_net.inputs)
        print('------NETWORK END-----')
        return retval, m.params


def basic_jerk_model(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, obj_pic_dims = None, scale_down_height = None, scale_down_width = None, add_depth_gaussian = False, include_pose = False, num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
#    with tf.device('/gpu:%d' % gpu_id):
	batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
	long_len = inputs['object_data'].get_shape().as_list()[1]
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, store_jerk = True, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file, scale_down_height = scale_down_height, scale_down_width = scale_down_width, add_depth_gaussian = add_depth_gaussian)

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

	num_encode_together = cfg.get('encode_together_depth')
	if num_encode_together:
		print('Encoding together!')
		together_input = tf.concat(encoded_input, axis = 3)
		encoded_input = feedforward_conv_loop(together_input, m, cfg, desc = 'encode_together', bypass_nodes = size_1_encoding_before_concat, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False)[-1:]	


	#flatten and concat

	flattened_input = [tf.reshape(enc_in, [batch_size, -1]) for enc_in in encoded_input]
	flattened_input.append(tf.reshape(inputs['object_data_seen_1d'], [batch_size, -1]))
	flattened_input.append(tf.reshape(inputs['actions_no_pos'], [batch_size, -1]))
	flattened_input.append(tf.reshape(inputs['depth_seen'], [batch_size, -1]))

	assert len(flattened_input[0].get_shape().as_list()) == 2
	concat_input = tf.concat(flattened_input, axis = 1)

	pred = hidden_loop_with_bypasses(concat_input, m, cfg, reuse_weights = False, train = kwargs['train'])
	pred_shape = pred.get_shape().as_list()
#	pred_shape = base_net.inputs['object_data_future'].get_shape().as_list()
#	if not include_pose:
#		pred_shape[3] = 2
#	print('num classes: ' + str(num_classes))
	if num_classes is not None:
		pred_shape.append(int(num_classes))
		pred_shape[1] = int(pred_shape[1] / num_classes)
		pred = tf.reshape(pred, pred_shape)
        retval = {'pred' : pred}
        retval.update(base_net.inputs)
        return retval, m.params


def correlation(x, y):
        x = tf.reshape(x, (-1,))
        y = tf.reshape(y, (-1,))
        n = tf.cast(tf.shape(x)[0], tf.float32)
        x_sum = tf.reduce_sum(x)
        y_sum = tf.reduce_sum(y)
        xy_sum = tf.reduce_sum(tf.multiply(x, y))
        x2_sum = tf.reduce_sum(tf.pow(x, 2))
        y2_sum = tf.reduce_sum(tf.pow(y, 2))
        numerator = tf.scalar_mul(n, xy_sum) - tf.scalar_mul(x_sum, y_sum)
        denominator = tf.sqrt(tf.scalar_mul(tf.scalar_mul(n, x2_sum) - tf.pow(x_sum, 2),
                                        tf.scalar_mul(n, y2_sum) - tf.pow(y_sum, 2)))
        corr = tf.truediv(numerator, denominator)
        return corr



def correlation_jerk_loss(outputs, l2_coef = 1.):
        pred = outputs['pred']
        tv = outputs['jerk']
        n_entries = np.prod(tv.get_shape().as_list())
        return l2_coef * tf.nn.l2_loss(pred - tv) / n_entries - correlation(pred, tv) + 1



def discretize(in_tensor, min_value, max_value, num_classes):
	assert in_tensor.dtype == tf.float32
	assert num_classes <= 256 #just making a little assumption here
	shifted_tensor = tf.maximum(tf.minimum(in_tensor, max_value), min_value)
	shifted_tensor = (in_tensor - min_value) / (max_value - min_value) * (num_classes - 1)
	discrete_tensor = tf.cast(shifted_tensor, tf.uint8)
	one_hotted = tf.one_hot(discrete_tensor, depth = num_classes)
	return one_hotted

def discretized_loss(outputs, num_classes = 40, min_value = -.5, max_value = .5):
	pred = outputs['pred']
	jerk = outputs['jerk']
	disc_jerk = discretize(jerk, min_value = min_value, max_value = max_value, num_classes = num_classes)
	print('pred and jerk')
	print((pred, disc_jerk))
#	batch_size = pred.get_shape().as_list()[0]
	cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels = disc_jerk, logits = pred)
	return tf.reduce_mean(cross_ent)

def softmax_cross_entropy_loss_per_pixel(outputs, gpu_id = 0, eps = 0.01, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
        labels = tf.cast(outputs['depths_raw'][:,-1,:,:,0], tf.int32) # only predict the coarsest channel
        logits = outputs['pred']
        weight = tf.abs(outputs['jerk_map']) + eps

        undersample = True
        if undersample:
            mask = tf.norm(outputs['jerk'], ord='euclidean', axis=1)
            mask = tf.cast(tf.greater(mask, 0.05909), tf.float32)
            mask = tf.reshape(mask, [mask.get_shape().as_list()[0], 1, 1, 1])
        else:
            mask = 1

        loss = tf.reduce_mean(tf.expand_dims(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), axis=3) * weight * mask)
        return [loss]

def softmax_cross_entropy_loss_with_bins(outputs, bin_data_file, 
        gpu_id = 0, clip_weight=None, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
        gt = outputs['jerk']
        # bin ground truth into n-bins
        with open(bin_data_file) as f:
            bin_data = cPickle.load(f)
        # upper bound of bin
        bins = bin_data['bins']
        # weighting of bin
        w = bin_data['weights']
        assert len(w) == len(bins) + 1, 'len(weights) != len(bins) + 1'
        w = tf.cast(w, tf.float32)
        labels = []
        for i in range(len(bins) + 1):
            if i == 0:
                label = tf.less(gt, bins[i])
            elif i == len(bins):
                label = tf.greater(gt, bins[i-1])
            else:
                label = tf.logical_and(tf.greater(gt, bins[i-1]), \
                        tf.less(gt, bins[i]))
            labels.append(label)
        labels = tf.stack(labels, axis=2)
        labels = tf.cast(labels, tf.float32)
        if clip_weight is not None:
            w = tf.minimum(w, clip_weight)
        w = labels * tf.expand_dims(tf.expand_dims(w, axis=0), axis=0)
        w = tf.reduce_sum(w, axis=2)
        pred = tf.cast(outputs['pred'], tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=pred)
        return tf.reduce_mean(loss)

def parallel_reduce_mean(losses, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for i, loss in enumerate(losses):
            losses[i] = tf.reduce_mean(loss)
        print(losses)
        return losses

class ParallelClipOptimizer(object):
    def __init__(self, optimizer_class, clip=True, gpu_offset=0, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip
        self.gpu_offset = gpu_offset

    def compute_gradients(self, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(*args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                   for grad, var in gvs if grad is not None]
        return gvs

    def minimize(self, losses, global_step):
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            grads_and_vars = []
            if isinstance(losses, list):
                for i, loss in enumerate(losses):
                    with tf.device('/gpu:%d' % i + self.gpu_offset):
                        with tf.name_scope('gpu_' + str(i + self.gpu_offset)) \
                                as gpu_scope:
                            grads_and_vars.append(self.compute_gradients(loss))
                            #tf.get_variable_scope().reuse_variables()
                grads_and_vars = self.average_gradients(grads_and_vars)
            else:
                with tf.device('/gpu:%d' % self.gpu_offset):
                    with tf.name_scope('gpu_' + str(self.gpu_offset)) as gpu_scope:
                        grads_and_vars = self.compute_gradients(losses)
            return self._optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)

    def average_gradients(self, all_grads_and_vars):
        average_grads_and_vars = []
        for grads_and_vars in zip(*all_grads_and_vars):
            grads = []
            for g, _ in grads_and_vars:
                grads.append(tf.expand_dims(g, axis=0))
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            # all variables are the same so we just use the first gpu variables
            var = grads_and_vars[0][1]
            grad_and_var = (grad, var)
            average_grads_and_vars.append(grad_and_var)
        return average_grads_and_vars


def gen_cfg_short_jerk(num_filters_before_concat = 24, num_filters_after_concat = 34, num_filters_together = 34, encode_depth = 2, encode_size = 7, hidden_depth = 3, hidden_num_features = 250):
	cfg = {'size_1_before_concat_depth' : 1, 'size_2_before_concat_depth' : 0, 'encode_depth' : encode_depth, 'hidden_depth' : hidden_depth, 'hidden' : {}, 'encode' : {}}
	cfg['size_1_before_concat'] = {
                1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : num_filters_before_concat}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        }
	for i in range(1, encode_depth + 1):
		cfg['encode'][i] = {'conv' : {'filter_size' : encode_size, 'stride' : 2, 'num_filters' : num_filters_after_concat}}
		if i > 1:
			cfg['encode'][i]['bypass'] = 0
	cfg['encode_together_depth'] = 1
	cfg['encode_together'] = {
                1 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : num_filters_together}, 'bypass' : 0}
        }
	for i in range(1, hidden_depth):
		cfg['hidden'][i] = {'num_features' : hidden_num_features, 'dropout' : .75}
	cfg['hidden'][hidden_depth] = {'num_features' : 3, 'activation' : 'identity'}
	return cfg






cfg_alt_short_jerk = {
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
        'encode_together_depth' : 1,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 34}, 'bypass' : 0}
        },

        'hidden_depth' : 3,
        'hidden' : {
                1: {'num_features' : 250, 'dropout' : .75},
                2 : {'num_features' : 250, 'dropout' : .75},
                3 : {'num_features' : 3, 'activation' : 'identity'}
        }
}

def cfg_bypass_jerk():
    return {
        'size_1_before_concat_depth' : 0,

        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                    'bypass' : 0},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                    'bypass' : 0},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_together_depth' : 2,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 128},
                    'bypass' : 0},
                2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 128},
                    'bypass' : 0},
                    #, 'bypass' : 0}
        },
        'hidden_depth': 0,

        'deconv_depth': 3,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                'bypass' : 0},
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                'bypass' : 0},
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                'bypass' : 0},
        }
}

def cfg_sym_jerk():
    return {
        'size_1_before_concat_depth' : 0,

        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128}},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128}},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_together_depth' : 2,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256}},
                2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256}},
                    #, 'bypass' : 0}
        },
        'hidden_depth': 0,

        'deconv_depth': 3,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}, 
                'bypass' : 3},
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                'bypass' : 2},
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                'bypass' : 1},
        }
}

def cfg_map_jerk():
    return {
        'size_1_before_concat_depth' : 0,

        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_together_depth' : 2,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256}},
                2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256}},
                    #, 'bypass' : 0}
        },
        'hidden_depth': 0,

        'deconv_depth': 3,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}},
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}},
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}},
        }
}

def cfg_res_jerk():
    return {
        'size_1_before_concat_depth' : 0,

        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32}},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_together_depth' : 3,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 7, 'stride' : 1, 'num_filters' : 32}},
                2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256}},
                    #, 'bypass' : 0}
        },
        'hidden_depth': 0,
        'deconv_depth': 0,
}


def cfg_class_jerk(num_classes_per_dim):
	return {
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
        'encode_together_depth' : 1,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 1, 'stride' : 1, 'num_filters' : 34}, 'bypass' : 0}
        },

        'hidden_depth' : 3,
        'hidden' : {
                1: {'num_features' : 250, 'dropout' : .75},
                2 : {'num_features' : 250, 'dropout' : .75},
                3 : {'num_features' : 3 * num_classes_per_dim, 'activation' : 'identity'}
        }
}

def cfg_class_jerk_bench(num_classes_per_dim):
	return {
	'hidden_depth' : 3,
	'hidden' : {
		1 : {'num_features' : 1000, 'dropout' : .75},
		2 : {'num_features' : 1000, 'dropout' : .75},
		3 : {'num_features' : 3 * num_classes_per_dim, 'activation' : 'identity'}
	}
}

def gen_cfg_jerk_bench(depth, width, drop_keep, num_classes_per_dim = 1):
	cfg = {'hidden_depth' : depth, 'hidden' : {}}
	for i in range(1, depth):
		cfg['hidden'][i] = {'num_features' : width, 'dropout' : drop_keep}
	cfg['hidden'][depth] = {'num_features' : 3 * num_classes_per_dim, 'activation' : 'identity'}
	return cfg






