'''
Policy and intrinsic reward models.
'''

import numpy as np
import tensorflow as tf
from curiosity.models.model_building_blocks import ConvNetwithBypasses
from curiosity.models import explicit_future_prediction_base as fp_base
from curiosity.models import jerk_models

import distutils.version
use_tf1 = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

def tf_concat(list_of_tensors, axis = 0):
    if use_tf1:
        return tf.concat(list_of_tensors, axis)
    return tf.concat(axis, list_of_tensors)

#TODO replace all these makeshift helpers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None, reuse_weights = False):
    with tf.variable_scope(name, reuse = reuse_weights):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b, w, b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b, w, b


class UniformActionSampler:
	def __init__(self, cfg):
		self.action_dim = cfg['uncertainty_model']['action_dim']
		self.num_actions = cfg['uncertainty_model']['n_action_samples']
		self.rng = np.random.RandomState(cfg['seed'])

	def sample_actions(self):
		return self.rng.uniform(-1., 1., [self.num_actions, self.action_dim])

def postprocess_depths(depths):
	'''
		Assumes depths is of shape [batch_size, time_number, height, width, 3]
	'''
	depths = tf.cast(depths, tf.float32)
	depths = (depths[:,:,:,:,0:1] * 256. + depths[:,:,:,:,1:2] + \
	        depths[:,:,:,:,2:3] / 256.0) / 1000.0 
	depths /= 4. # normalization
	return depths

def postprocess_std(in_node):
	in_node = tf.cast(in_node, tf.float32)
	in_node = in_node / 255.
	return in_node


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
    cfs0 = None

    if bypass_nodes is None:
        bypass_nodes = [m.output]

    for i in range(1, deconv_depth + 1):
        with tf.variable_scope(desc + str(i)) as scope:
            if reuse_weights:
                scope.reuse_variables()

            bypass = cfg[desc][i].get('bypass')
            if bypass is not None:
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
                            if no_nonlinearity_end and i == encode_depth:
                                m.conv(nf, cfs, cs, activation = None)
                            else:
                                my_activation = cfg[desc][i].get('nonlinearity')
                                if my_activation is None:
                                        my_activation = 'relu'
                                m.conv(nf, cfs, cs, activation = my_activation)
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


def action_softmax_loss(prediction, tv, num_classes = 21, min_value = -1., max_value = 1.):
	#get into the right shape
	tv_shape = tv.get_shape().as_list()
	pred_shape = prediction.get_shape().as_list()
	print(tv_shape)
	print(pred_shape)
	assert len(tv_shape) == 2 and tv_shape[1] * num_classes == pred_shape[1], (len(tv_shape), tv_shape[1] * num_classes, pred_shape[1])
	pred = tf.reshape(prediction, [-1, tv_shape[1], num_classes])
	#discretize tv
	tv = float(num_classes) * (tv - min_value) / (max_value - min_value)
	tv = tf.cast(tv, tf.int32)
	loss_per_example = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels = tv, logits = pred))
	loss = tf.reduce_mean(loss_per_example)
	return loss_per_example, loss
	

def softmax_cross_entropy_loss_vel_one(outputs, tv, gpu_id = 0, eps = 0.0,
        min_value = -1.0, max_value = 1.0, num_classes=256,
        segmented_jerk=True, **kwargs):
    #with tf.device('/gpu:%d' % gpu_id):
        undersample = False
        if undersample:
            thres = 0.5412
            mask = tf.norm(outputs['jerk_all'], ord='euclidean', axis=2)
            mask = tf.cast(tf.logical_or(tf.greater(mask[:,0], thres),
                tf.greater(mask[:,1], thres)), tf.float32)
            mask = tf.reshape(mask, [mask.get_shape().as_list()[0], 1, 1, 1])
        else:
            mask = 1
        shape = outputs['pred_next_vel_1'].get_shape().as_list()
        assert shape[3] / 3 == num_classes

        losses = []
        # next image losses
        logits = outputs['next_images'][1][0]
        logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
        labels = tf.cast(tv, tf.int32)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits) * mask)
        losses.append(loss)
        assert len(losses) == 1, ('loss length: %d' % len(losses))

        losses = tf.stack(losses)
        return tf.reduce_mean(losses)

default_damian_cfg = jerk_models.cfg_mom_complete_bypass(768, use_segmentation=False,
            method='concat', nonlin='relu')

default_damian_cfg.update({'state_shape' : [2, 128, 170, 3], 'action_shape' : [2, 8]})


class DamianModel:
	def __init__(self, cfg = None, time_seen = 3):
		self.s_i = x = tf.placeholder(tf.float32, [1] + cfg['state_shape'])
		self.s_f = s_f = tf.placeholder(tf.float32, [1] + cfg['state_shape'])
		self.objects = objects = tf.placeholder(tf.float32, [1] + cfg['state_shape'])
		self.action = action = tf.placeholder(tf.float32, [1, 2, cfg['action_dim']])
		self.action_id = action_id = tf.placeholder(tf.int32, [1, 2])
		bs = tf.to_float(tf.shape(self.s_i)[0])
		final_img_unsqueezed = self.s_f[:, 1:]
		depths = tf_concat([self.s_i, final_img_unsqueezed], 1)
		inputs = replace_base(depths, objects, action, action_id)
		self.processed_input = inputs
		for k, inpt in inputs.iteritems():
			print(k)
			print(inpt)
		#then gotta postprocess things to be of the right form, get a thing called inputs out of it
		self.model_results, _ = mom_complete(inputs, cfg = cfg, time_seen = time_seen)
		self.pred = self.model_results['next_images']
		self.tv = inputs['tv']
		self.loss = softmax_cross_entropy_loss_vel_one(self.model_results, self.tv, segmented_jerk = False, buckets = 255)
		
		
def replace_base(depths, objects, action, action_id):
        inputs = {'depths' : depths, 'objects' : objects, 'actions' : action}
        rinputs = {}
        for k in inputs:
            if k in ['depths', 'objects']:
                rinputs[k] = tf.pad(inputs[k],
                        [[0,0], [0,0], [0,0], [3,3], [0,0]], "CONSTANT")
                # RESIZING IMAGES
                rinputs[k] = tf.unstack(rinputs[k], axis=1)
                for i, _ in enumerate(rinputs[k]):
                    rinputs[k][i] = tf.image.resize_images(rinputs[k][i], [64, 88])
                rinputs[k] = tf.stack(rinputs[k], axis=1)
            else:
                rinputs[k] = inputs[k]
	objects = tf.cast(rinputs['objects'], tf.int32)
	shape = objects.get_shape().as_list()
	objects = tf.unstack(objects, axis=len(shape)-1)
	objects = objects[0] * (256**2) + objects[1] * 256 + objects[2]
	action_id = tf.expand_dims(action_id, 2)
	action_id = tf.cast(tf.reshape(tf.tile(action_id, [1, 1, shape[2] * shape[3]]), shape[:-1]), tf.int32)
        actions = tf.cast(tf.equal(objects, action_id), tf.float32)
        actions = tf.tile(tf.expand_dims(actions, axis = 4), [1, 1, 1, 1, 6])
        actions *= tf.expand_dims(tf.expand_dims(action[:, :, 2:], 2), 2)
        ego_motion = tf.expand_dims(tf.expand_dims(action[:, :, :2], 2), 2)
        ego_motion = tf.tile(ego_motion, [1, 1, shape[2], shape[3], 1])
	action_map = tf_concat([actions, ego_motion], -1)
	action_map = tf.expand_dims(action_map, -1)
	inputs = {'actions_map' : action_map, 'depths' : postprocess_depths(rinputs['depths']), 'tv' : rinputs['depths'][:, -1] }
	return inputs 


def mom_complete(inputs, cfg = None, time_seen = None, normalization_method = None,
        stats_file = None, obj_pic_dims = None, scale_down_height = None,
        scale_down_width = None, add_depth_gaussian = False, add_gaussians = False,
        include_pose = False, store_jerk = True, use_projection = False, 
        num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
        #print('------NETWORK START-----')
        #with tf.device('/gpu:%d' % gpu_id):
        # rescale inputs to be divisible by 8
        #rinputs = {}
        #for k in inputs:
        #    if k in ['depths', 'objects', 'vels', 'accs', 'jerks',
        #            'vels_curr', 'accs_curr', 'actions_map', 'segmentation_map']:
        #        rinputs[k] = tf.pad(inputs[k],
        #                [[0,0], [0,0], [0,0], [3,3], [0,0]], "CONSTANT")
        #        # RESIZING IMAGES
        #        rinputs[k] = tf.unstack(rinputs[k], axis=1)
        #        for i, _ in enumerate(rinputs[k]):
        #            rinputs[k][i] = tf.image.resize_images(rinputs[k][i], [64, 88])
        #        rinputs[k] = tf.stack(rinputs[k], axis=1)
        #    else:
        #        rinputs[k] = inputs[k]
       # preprocess input data
        batch_size, time_seen, height, width = \
                inputs['depths'].get_shape().as_list()[:4]
        time_seen -= 1
        long_len = time_seen + 1
        #base_net = fp_base.ShortLongFuturePredictionBase(
        #        rinputs, store_jerk = store_jerk,
        #        normalization_method = normalization_method,
        #        time_seen = time_seen, stats_file = stats_file,
        #        scale_down_height = scale_down_height,
        #        scale_down_width = scale_down_width,
        #        add_depth_gaussian = add_depth_gaussian,
        #        add_gaussians = add_gaussians,
        #        get_hacky_segmentation_map = True,
        #        get_actions_map = True)
        #inputs = base_net.inputs

        # init network
        m = ConvNetwithBypasses(**kwargs)

        # encode per time step
        main_attributes = ['depths']
        main_input_per_time = [tf_concat([tf.cast(inputs[nm][:, t], tf.float32) \
                for nm in main_attributes], axis = 3) for t in range(time_seen)]

        # init projection matrix
        if use_projection:
            print('Using PROJECTION')
            with tf.variable_scope('projection'):
                P = tf.get_variable(name='P',
                        initializer=tf.eye(4),
                        #shape=[4, 4], 
                        dtype=tf.float32)
                
        # initial bypass
        bypass_nodes = [[b] for b in tf.unstack(inputs['depths'][:,:time_seen], axis=1)]

        # use projection
        if use_projection:
            for t in range(time_seen):
                main_input_per_time[t] = apply_projection(main_input_per_time[t], P)
                #bypass_nodes[t].append(main_input_per_time[t])

        # conditioning
        if 'use_segmentation' in cfg:
            use_segmentation = cfg['use_segmentation']
        else:
            use_segmentation = False

        print('Using ACTION CONDITIONING')
        cond_attributes = ['actions_map']
        if use_segmentation:
            print('Using segmentations as conditioning')
            cond_attributes.append('segmentation_map')
        if 'cond_scale_factor' in cfg:
            scale_factor = cfg['cond_scale_factor']
        else:
            scale_factor = 1
        for att in cond_attributes:
            if att in ['actions_map']:
                inputs[att] = tf.reduce_sum(inputs[att], axis=-1, keep_dims=False)
            if att in ['segmentation_map']:
                inputs[att] = tf.reduce_sum(inputs[att], axis=-1, keep_dims=True)
            shape = inputs[att].get_shape().as_list()
            inputs[att] = tf.unstack(inputs[att], axis=1)
            for t, _ in enumerate(inputs[att]):
                inputs[att][t] = tf.image.resize_images(inputs[att][t],
                        [shape[2]/scale_factor, shape[3]/scale_factor],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            inputs[att] = tf.stack(inputs[att], axis=1)
        cond_input_per_time = [tf_concat([inputs[nm][:, t] \
                for nm in cond_attributes], axis = 3) for t in range(time_seen)]

        encoded_input_cond = []
        reuse_weights = False
        print('right before bug loop')
        for inpt in cond_input_per_time:
            print(inpt)
        for t in range(time_seen):
            enc, bypass_nodes[t] = feedforward_conv_loop(
                    cond_input_per_time[t], m, cfg, desc = 'cond_encode',
                    bypass_nodes = bypass_nodes[t], reuse_weights = reuse_weights,
                    batch_normalize = False, no_nonlinearity_end = False,
                    do_print=(not reuse_weights), return_bypass = True)
            encoded_input_cond.append(enc[-1])
            reuse_weights = True

        # main
        encoded_input_main = []
        reuse_weights = False
        for t in range(time_seen):
                enc, bypass_nodes[t] = feedforward_conv_loop(
                        main_input_per_time[t], m, cfg, desc = 'main_encode',
                        bypass_nodes = bypass_nodes[t], reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                encoded_input_main.append(enc[-1])
                reuse_weights = True

        # calculate moments
        bypass_nodes = [bypass_nodes]
        moments = [encoded_input_main]
        reuse_weights = False
        assert time_seen-1 > 0, ('len(time_seen) = 0')
        for i, mom in enumerate(range(time_seen-1, 0, -1)):
            sub_bypass_nodes = []
            for t in range(mom):
                bn = []
                for node in bypass_nodes[i][t]:
                    bn.append(node)
                sub_bypass_nodes.append(bn)
            bypass_nodes.append(sub_bypass_nodes)

            sub_moments = []
            for t in range(mom):
                sm = moments[i]
                if cfg['combine_moments'] == 'minus':
                    print('Using MINUS')
                    enc = sm[t+1] - sm[t]
                elif cfg['combine_moments'] == 'concat':
                    print('Using CONCAT')
                    enc = tf_concat([sm[t+1], sm[t]], axis=3)
                    enc, bypass_nodes[i+1][t] = feedforward_conv_loop(
                            enc, m, cfg, desc = 'combine_moments_encode',
                            bypass_nodes = bypass_nodes[i+1][t], 
                            reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print=(not reuse_weights), return_bypass = True,
                            sub_bypass = i)
                    enc = enc[-1]
                enc, bypass_nodes[i+1][t] = feedforward_conv_loop(
                        enc, m, cfg, desc = 'moments_encode',
                        bypass_nodes = bypass_nodes[i+1][t], 
                        reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True,
                        sub_bypass = i)
                sub_moments.append(enc[-1])
                reuse_weights = True
            moments.append(sub_moments)

        # concat moments, main and cond
        currents = []
        reuse_weights = False
        for i, moment in enumerate(moments):
            sub_currents = []
            for t, _ in enumerate(moment):
                enc = tf_concat([moment[t], 
                    encoded_input_main[t+i], #TODO first moments are main inputs already!
                    encoded_input_cond[t+i]], axis=3)
                enc, bypass_nodes[i][t] = feedforward_conv_loop(
                        enc, m, cfg, desc = 'moments_main_cond_encode',
                        bypass_nodes = bypass_nodes[i][t], reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True,
                        sub_bypass = i)
                sub_currents.append(enc[-1])
                reuse_weights = True
            currents.append(sub_currents)

        # predict next moments via residuals (delta moments)
        next_moments = []
        delta_moments = []
        reuse_weights = False
        for i, current in enumerate(currents):
            next_moment = []
            delta_moment = []
            for t, _ in enumerate(current):
                dm, bypass_nodes[i][t] = feedforward_conv_loop(
                        current[t], m, cfg, desc = 'delta_moments_encode',
                        bypass_nodes = bypass_nodes[i][t], reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True,
                        sub_bypass = i)
                if cfg['combine_delta'] == 'plus':
                    print('Using PLUS')
                    nm = current[t] + dm[-1]
                elif cfg['combine_delta'] == 'concat':
                    print('Using CONCAT')
                    nm = tf_concat([current[t], dm[-1]], axis=3)
                    nm, bypass_nodes[i][t] = feedforward_conv_loop(
                            nm, m, cfg, desc = 'combine_delta_encode',
                            bypass_nodes = bypass_nodes[i][t], 
                            reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print=(not reuse_weights), return_bypass = True,
                            sub_bypass = i)
                    nm = nm[-1]
                else:
                    raise KeyError('Unknown combine_delta')
                reuse_weights = True
                delta_moment.append(dm[-1])
                next_moment.append(nm)
            next_moments.append(next_moment)
            delta_moments.append(delta_moment)

        # concat next moments and main and reconstruct
        nexts = []
	reuse_weights = False
	for i, moment in enumerate(next_moments):
            sub_nexts = []
	    for t, _ in enumerate(moment):
                # TODO: first moments are main inputs already!
                # -> no need to concat for i == 0
                # TODO: Higher moment reconstruction needs additional layers
                # to match dimensions -> depth + vel + acc to next vel 
                # vs depth + vel to next depth -> only vel possible so far!
		enc = tf_concat([moment[t], encoded_input_main[t+i]], axis=3)
		enc, bypass_nodes[i][t] = feedforward_conv_loop(
			enc, m, cfg, desc = 'next_main_encode',
			bypass_nodes = bypass_nodes[i][t], reuse_weights = reuse_weights,
			batch_normalize = False, no_nonlinearity_end = False,
			do_print=(not reuse_weights), return_bypass = True,
                        sub_bypass = i)
		reuse_weights = True
                sub_nexts.append(enc[-1])
            nexts.append(sub_nexts)

        # Deconvolution
        num_deconv = cfg.get('deconv_depth')
        reuse_weights = False
        if num_deconv:
            for i, moment in enumerate(moments):
                for t, _ in enumerate(moment):
                    enc, bypass_nodes[i][t] = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes[i][t], 
                            reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True,
                            sub_bypass = i)
                    moment[t] = enc[-1]
                    reuse_weights = True
            for i, moment in enumerate(next_moments):
                for t, _ in enumerate(moment):
                    enc, bypass_nodes[i][t] = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes[i][t], 
                            reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True,
                            sub_bypass = i)
                    moment[t] = enc[-1]
                    reuse_weights = True
            for i, moment in enumerate(delta_moments):
                for t, _ in enumerate(moment):
                    enc, bypass_nodes[i][t] = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes[i][t], 
                            reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True,
                            sub_bypass = i)
                    moment[t] = enc[-1]
                    reuse_weights = True
            for i, moment in enumerate(nexts):
                for t, _ in enumerate(moment):
                    enc, bypass_nodes[i][t] = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes[i][t], 
                            reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True,
                            sub_bypass = i)
                    moment[t] = enc[-1]
                    reuse_weights = True
        retval = {
                'pred_vel_1': moments[1][0],
                'pred_delta_vel_1': delta_moments[1][0],
                'pred_next_vel_1': next_moments[1][0],
                'pred_next_img_1': nexts[1][0],
                #'pred_next_vel_2': next_moments[0][1],
                'bypasses': bypass_nodes,
                'moments': moments,
                'delta_moments': delta_moments,
                'next_moments': next_moments,
                'next_images': nexts
                }
        retval.update(inputs)
        print('------NETWORK END-----')
        print('------BYPASSES-------')
        for i, node in enumerate(bypass_nodes[1][0]):
            print(i, bypass_nodes[1][0][i])
        for i, mn in enumerate(bypass_nodes):
            for j, tn in enumerate(mn):
                print('------LENGTH------', i, j, len(tn))
                #for k, bn in enumerate(tn):
                #    print(i, j, k, bn)
        print(len(bypass_nodes))
        return retval, m.params







def hidden_loop_with_bypasses(input_node, m, cfg, nodes_for_bypass = [], stddev = .01, reuse_weights = False, activation = 'relu', train = True):
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
                        my_activation = cfg['hidden'][i].get('activation')
                        if my_activation is None:
                                my_activation = activation
                        if train:
                                my_dropout = cfg['hidden'][i].get('dropout')
                        else:
                                my_dropout = None
                        m.fc(nf, init = 'xavier', activation = my_activation, bias = .01, stddev = stddev, dropout = my_dropout)
                        nodes_for_bypass.append(m.output)
        return m.output


def flatten_append_unflatten(start_state, action, cfg, m):
    x = flatten(start_state)
    action = flatten(action)
    joined = tf_concat([x, action], 1)
    x = hidden_loop_with_bypasses(joined, m, cfg['mlp'], reuse_weights = False, train = True)
    reshape_dims = cfg['reshape_dims']
    # assert np.prod(reshape_dims) == tf.shape(joined)[-1],  (np.prod(reshape_dims), tf.shape(joined)[-1])
    return tf.reshape(x, [-1] + reshape_dims)


class DepthFuturePredictionWorldModel():
	def __init__(self, cfg, action_state_join_model = flatten_append_unflatten):
		print('Warning! dropout train/test not currently being handled.')
		with tf.variable_scope('wm'):
			#state shape gives the state of one shape. The 'states' variable has an extra timestep, which it cuts up into the given and future states.
			states_shape = list(cfg['state_shape'])
			states_shape[0] += 1
			#Knowing the batch size is not truly needed, but until we need this to be adaptive, might as well keep it
			#The fix involves getting shape information to deconv
			bs = cfg['batch_size']
			self.states = tf.placeholder(tf.float32, [bs] + states_shape)
			self.s_i = x = self.states[:, :-1]
			self.s_f = s_f = self.states[:, 1:]
			self.action = tf.placeholder(tf.float32, [bs] + cfg['action_shape'])
			#convert from 3-channel encoding
			self.processed_input = x = postprocess_depths(x)

			s_f = postprocess_depths(s_f)
			#flatten time dim
			x = tf_concat([x[:, i] for i in range(cfg['state_shape'][0])], 3)
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
			diff = self.pred - self.tv
			diff = flatten(diff)
			per_sample_norm = cfg.get('per_sample_normalization')
			if per_sample_norm == 'reduce_mean':
				self.loss_per_example = tf.reduce_mean(diff * diff / 2., axis = 1)
			else:	
				self.loss_per_example = tf.reduce_sum(diff * diff / 2., axis = 1)
			self.loss = tf.reduce_mean(self.loss_per_example)
			#self.loss = tf.nn.l2_loss(self.tv - self.pred) #bs #(bs * np.prod(cfg['state_shape']))



sample_depth_future_cfg = {
	'state_shape' : [2, 64, 64, 3],
	'action_shape' : [2, 8],
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
	'action_shape' : [2, 8],

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
			1 : {'conv' : {'filter_size' : 5, 'stride' : 2, 'num_filters' : 20}},
			2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 20}},
			3 : {'conv' : {'filter_size' : 5, 'stride' : 2, 'num_filters' : 20}},
			4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 10}},
			5 : {'conv' : {'filter_size' : 5, 'stride' : 2, 'num_filters' : 5}},
		}
	},

	'deconv' : {
		'deconv_depth' : 5,

		'deconv' : {
			1 : {'deconv' : {'filter_size' : 5, 'stride' : 2, 'num_filters' : 20}, 'bypass' : 4},
			2 : {'deconv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 20}, 'bypass' : 3},
			3 : {'deconv' : {'filter_size' : 5, 'stride' : 2, 'num_filters' : 20}, 'bypass' : 2},
			4 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 10}, 'bypass' : 1},
			5 : {'deconv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 1},  'bypass' : 0}
		}


	}


}


class ActionModel(object):
    def __init__(self, cfg):
        states_shape = list(cfg['state_shape'])
        states_shape[0] += 1
        self.states = tf.placeholder(tf.float32, [None] + states_shape)
        self.s_i = s_i = self.states[:, :-1]
        self.s_f = s_f = self.states[:, 1:]
        self.action = tf.placeholder(tf.float32, [None] + cfg['action_shape'])
        self.action_post = tf.placeholder(tf.float32, [None] + cfg['action_shape'])
        last_action = self.action[:, 0]
        tv_action = self.action_post[:, -1]


        #encode
        s_i = tf_concat([s_i[:, i] for i in range(cfg['state_shape'][0])], 3)
        s_f = tf_concat([s_f[:, i] for i in range(cfg['state_shape'][0])], 3)

        s_i = postprocess_std(s_i)
        s_f = postprocess_std(s_f)

        m = ConvNetwithBypasses()
        with tf.variable_scope('encode_model'):
            s_i = feedforward_conv_loop(s_i, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False)[-1]
            s_f = feedforward_conv_loop(s_f, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = True, batch_normalize = False, no_nonlinearity_end = False)[-1]
        
        #action mlp
        enc_i_flat = flatten(s_i)
        enc_f_flat = flatten(s_f)
        if cfg['action_model'].get('include_last_action'):
            to_concat = [enc_i_flat, enc_f_flat, last_action]
        else:
            to_concat = [enc_i_flat, enc_f_flat]
        enc_in = tf_concat(to_concat, 1)
        self.act_pred = hidden_loop_with_bypasses(enc_in, m, cfg['action_model']['mlp'], reuse_weights = False, train = True)

        #loss
        loss_factor = cfg['action_model'].get('loss_factor', 1.)
        self.act_loss = tf.nn.l2_loss(self.act_pred - tv_action) * loss_factor




class LatentSpaceWorldModel(object):
    def __init__(self, cfg):
	#states shape has one more timestep, because we have given and future times, shoved into gpu once, and then we cut it up
	states_shape = list(cfg['state_shape'])
	states_shape[0] += 1
	self.states = tf.placeholder(tf.float32, [None] + states_shape)
        self.s_i = s_i = self.states[:, :-1]
        self.s_f = s_f = self.states[:, 1:]
        self.action = tf.placeholder(tf.float32, [None] + cfg['action_shape'])
        self.encode_var_list = []
	self.action_post = tf.placeholder(tf.float32, [None] + cfg['action_shape'])

        #flatten out time dim
        s_i = tf_concat([s_i[:, i] for i in range(cfg['state_shape'][0])], 3)
        s_f = tf_concat([s_f[:, i] for i in range(cfg['state_shape'][0])], 3)

	s_i = postprocess_std(s_i)
	s_f = postprocess_std(s_f)


        m = ConvNetwithBypasses()

        with tf.variable_scope('encode_model'):
            s_i = feedforward_conv_loop(s_i, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False)[-1]
            s_f = feedforward_conv_loop(s_f, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = True, batch_normalize = False, no_nonlinearity_end = False)[-1]

        self.encoding_i = s_i
        self.encoding_f = s_f

        enc_i_flat = flatten(s_i)
        enc_f_flat = flatten(s_f)
        act_flat = flatten(self.action)

	act_loss_factor = cfg['action_model'].get('loss_factor', 1.)
	fut_loss_factor = cfg['future_model'].get('loss_factor', 1.)

        #action model time
        with tf.variable_scope('action_model'):
            loss_type = cfg['action_model'].get('loss_type', 'both_l2')
            include_prev_action = cfg['action_model'].get('include_previous_action', False)
            to_concat = [enc_i_flat, enc_f_flat]
            if include_prev_action:
                print('including previous action!')
                to_concat.append(self.action[:, 0])
            encoded_concat = tf_concat(to_concat, 1)
            self.act_pred = hidden_loop_with_bypasses(encoded_concat, m, cfg['action_model']['mlp'], reuse_weights = False, train = True)
            if loss_type == 'both_l2':
                act_post_flat = flatten(self.action_post)
		diff = self.act_pred - act_post_flat
		self.act_loss_per_example = tf.reduce_sum(diff * diff, axis = 1) / 2. * act_loss_factor
                self.act_loss = tf.reduce_mean(self.act_loss_per_example)
            elif loss_type == 'one_l2':
                act_post_flat = self.action_post[:, -1]
		diff = self.act_pred - act_post_flat
		self.act_loss_per_example = tf.reduce_sum(diff * diff, axis = 1) / 2. * act_loss_factor
                self.act_loss = tf.reduce_mean(self.act_loss_per_example)
            elif loss_type == 'one_cat':
		print('cat')
		num_classes = cfg['action_model']['num_classes']
		act_post_flat = self.action_post[:, -1]
		self.act_loss_per_example, self.act_loss = action_softmax_loss(self.act_pred, act_post_flat, num_classes = num_classes)
            else:
                raise Exception('loss type not recognized!')

        #future model time
        enc_shape = enc_f_flat.get_shape().as_list()
        with tf.variable_scope('future_model'):
            fut_input = tf_concat([enc_i_flat, act_flat], 1)
            forward = hidden_loop_with_bypasses(fut_input, m, cfg['future_model']['mlp'], reuse_weights = False, train = True)
            if 'deconv' in cfg['future_model']:
                reshape_dims = cfg['future_model']['reshape_dims']
                forward = tf.reshape(forward, [-1] + reshape_dims)
                decoding = deconv_loop(
                                forward, m, cfg['future_model']['deconv'], desc='deconv',
                                bypass_nodes = [self.encoding_i], reuse_weights = False,
                                batch_normalize = False,
                                do_print = True)
                self.fut_pred = decoding[-1]
                tv_flat = flatten(self.encoding_f)
                # self.fut_loss = tf.nn.l2_loss(self.encoding_f - self.fut_pred)
            else:
                self.fut_pred = forward
                tv_flat = enc_f_flat
                # self.fut_loss = tf.nn.l2_loss(enc_f_flat - self.fut_pred)
            #different formula for l2 loss now because we need per-example details
            pred_flat = flatten(self.fut_pred)
            diff = pred_flat - tv_flat
            self.fut_loss_per_example = tf.reduce_sum(diff * diff, axis = 1) / 2. * fut_loss_factor
            self.fut_loss = tf.reduce_mean(self.fut_loss_per_example)
        self.act_var_list = [var for var in tf.global_variables() if 'action_model' in var.name]
        self.fut_var_list = [var for var in tf.global_variables() if 'future_model' in var.name]
        self.encode_var_list = [var for var in tf.global_variables() if 'encode_model' in var.name]

hourglass_latent_model_cfg = {
    'state_shape' : [2, 64, 64, 3],
    'action_shape' : [2, 8],
    'encode' : {
        'encode_depth' : 4,

        'encode' : {
            1: {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32}},
            2: {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32}},
            3: {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32}},
            4: {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}}
        }
    },

    'action_model' : {
        'mlp' : {
            'hidden_depth' : 2,
            'hidden' : {
                1: {'num_features' : 256},
                2: {'num_features' : 16, 'activation' : 'identity'}
            }
        }

    },

    'future_model' : {
        'mlp' : {
            'hidden_depth' : 2,
            'hidden' : {
                1: {'num_features' : 256},
                2: {'num_features' : 128, 'activation' : 'identity'}
            }

        },

        'reshape_dims' : [4, 4, 8],

        'deconv' : {
            'deconv_depth' : 3,

            'deconv' : {
                1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16}, 'bypass' : 0},
                2 : {'deconv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 16}, 'bypass' : 0},
                3 : {'deconv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}, 'bypass' : 0}
            }

        }


    }





}

mario_world_model_config = {
    'state_shape' : [2, 64, 64, 3],
    'action_shape' : [2, 8],
    'encode' : {
        'encode_depth' : 4,

        'encode' : {
            1: {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32}},
            2: {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32}},
            3: {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32}},
            4: {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32}}
        }
    },

    'action_model' : {
        'mlp' : {
            'hidden_depth' : 2,
            'hidden' : {
                1: {'num_features' : 256},
                2: {'num_features' : 16, 'activation' : 'identity'}
            }
        }

    },

    'future_model' : {
        'mlp' : {
            'hidden_depth' : 2,
            'hidden' : {
                1: {'num_features' : 512},
                2: {'num_features' : 512, 'activation' : 'identity'}
            }

        }


    }


}


class MixedUncertaintyModel:
	'''For both action and future uncertainty prediction, simultaneously, as separate predictions.
	Consider merging with UncertaintyModel, but right now that might look too messy. Want to leave that functionality alone.
	'''
	def __init__(self, cfg):
		with tf.variable_scope('uncertainty_model'):
			self.s_i = x = tf.placeholder(tf.float32, [None] + cfg['state_shape'])
			self.action_sample = ac = tf.placeholder(tf.float32, [None, cfg['action_dim']])
			self.true_act_loss = tf.placeholder(tf.float32, [None])
			self.true_fut_loss = tf.placeholder(tf.float32, [None])
			m = ConvNetwithBypasses()
			x = postprocess_depths(x)
			#concat temporal dims into channels
			x = tf_concat([x[:, i] for i in range(cfg['state_shape'][0])], 3)
			self.encoded = x = feedforward_conv_loop(x, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False)[-1]
			x = flatten(x)
			x = tf.cond(tf.equal(tf.shape(self.action_sample)[0], cfg['n_action_samples']), lambda : tf.tile(x, [cfg['n_action_samples'], 1]), lambda : x)
			fc_inputs = tf_concat([x, ac], 1)
			self.estimated_act_loss = hidden_loop_with_bypasses(fc_inputs, m, cfg['act_mlp'], reuse_weights = False, train = True)
			self.estimated_fut_loss = hidden_loop_with_bypasses(fc_inpits, m, cfg['fut_mlp'], reuse_weights = False, train = True)
			#TODO FINISH



class UncertaintyModel:
    def __init__(self, cfg):
	um_scope = cfg.get('scope_name', 'uncertainty_model')
        with tf.variable_scope(um_scope):
            self.s_i = x = tf.placeholder(tf.float32, [None] + cfg['state_shape'])
            self.action_sample = ac = tf.placeholder(tf.float32, [None, cfg['action_dim']])
            self.num_timesteps = cfg['state_shape'][0]
            self.true_loss = tr_loss = tf.placeholder(tf.float32, [None])
            m = ConvNetwithBypasses()
            x = postprocess_depths(x)
            #concatenate temporal dimension into channels
            x = tf_concat([x[:, i] for i in range(cfg['state_shape'][0])], 3)
            #encode
            self.encoded = x = feedforward_conv_loop(x, m, cfg['encode'], desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False)[-1]
            x = flatten(x)
            x = tf.cond(tf.equal(tf.shape(self.action_sample)[0], cfg['n_action_samples']), lambda : tf.tile(x, [cfg['n_action_samples'], 1]), lambda : x)
            # x = tf.tile(x, [cfg['n_action_samples'], 1])
            x = tf_concat([x, ac], 1)
            self.estimated_world_loss = x = hidden_loop_with_bypasses(x, m, cfg['mlp'], reuse_weights = False, train = True)
            x_tr = tf.transpose(x)
            heat = cfg.get('heat', 1.)
            x_tr /= heat
            prob = tf.nn.softmax(x_tr)
            log_prob = tf.nn.log_softmax(x_tr)
            self.entropy = - tf.reduce_sum(prob * log_prob)
            self.sample = categorical_sample(x_tr, cfg['n_action_samples'], one_hot = False)
            loss_factor = cfg.get('loss_factor', 1.)
            self.uncertainty_loss = tf.nn.l2_loss(self.estimated_world_loss - self.true_loss) * loss_factor
            self.state_descriptor = cfg['state_descriptor']
            self.just_random = False
            if 'just_random' in cfg:
                self.just_random = True
            	self.rng = np.random.RandomState(cfg['just_random'])

    def act(self, sess, action_sample, state, full_info = False):
        last_depths = state[self.state_descriptor][-1]#assuming this is always not None, as it should be getting an actual observation
        depths = state[self.state_descriptor][-2:]
        depths = [np.zeros(last_depths.shape, dtype = last_depths.dtype) if depth_t is None else depth_t for depth_t in depths]
        depths_batch = np.array([depths])
        if full_info:
            chosen_idx, entropy, estimated_world_loss = sess.run([self.sample, self.entropy, self.estimated_world_loss], 
							feed_dict = {self.s_i : depths_batch, self.action_sample : action_sample})
            chosen_idx = chosen_idx[0]
            if self.just_random:
                print('random act!')
                chosen_idx = self.rng.randint(len(action_sample))
            return action_sample[chosen_idx], entropy, estimated_world_loss
        chosen_idx = sess.run(self.sample, feed_dict = {self.s_i : depths_batch, self.action_sample : action_sample})[0]
        if self.just_random:
            print('random act!')
            chosen_idx = self.rng.randint(len(action_sample))
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

another_sample_cfg = {
	'uncertainty_model' : {
		'state_shape' : [2, 64, 64, 3],
		'action_dim' : 8,
		'n_action_samples' : 50,
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
		'mlp' : {
			'hidden_depth' : 2,
			'hidden' : {1 : {'num_features' : 20, 'dropout' : .75},
						2 : {'num_features' : 1, 'activation' : 'identity'}
			}		
		}
	},

	'world_model' : a_bigger_depth_future_config,

	'seed' : 0


}




default_damian_full_cfg = {
        'uncertainty_model' : {
                'state_shape' : [2, 128, 170, 3],
                'action_dim' : 8,
                'n_action_samples' : 50,
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
                'mlp' : {
                        'hidden_depth' : 2,
                        'hidden' : {1 : {'num_features' : 20, 'dropout' : .75},
                                                2 : {'num_features' : 1, 'activation' : 'identity'}
                        }
                }
        },



	'world_model' : default_damian_cfg,
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
