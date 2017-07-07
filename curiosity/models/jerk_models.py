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
        reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print = True, return_bypass=False, sub_bypass = None, use_3d = False):
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

            with tf.contrib.framework.arg_scope([m.deconv, m.deconv3d], 
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
                    if use_3d:
                        m.deconv3d(nf, cfs, cs, activation = None, 
                                fixed_output_shape=out_shape)
                    else:
                        m.deconv(nf, cfs, cs, activation = None,
                                fixed_output_shape=out_shape)
                else:
                    my_activation = cfg[desc][i].get('nonlinearity')
                    if my_activation is None:
                        my_activation = 'relu'
                    if use_3d:
                        m.deconv3d(nf, cfs, cs, activation = my_activation, 
                                fixed_output_shape=out_shape)
                    else:
                        m.deconv(nf, cfs, cs, activation = my_activation,
                                fixed_output_shape=out_shape)
                    if do_print:
                        print('deconv out:', m.output)
                    #TODO add print function
                    pool = cfg[desc][i].get('pool')
                    if pool:
                        pfs = pool['size']
                        ps = pool['stride']
                        if use_3d:
                            m.pool3d(pfs, ps)
                        else:
                            m.pool(pfs, ps)
                    deconv_nodes.append(m.output)
                    bypass_nodes.append(m.output)
    if return_bypass:
        return [deconv_nodes, bypass_nodes]
    return deconv_nodes

def feedforward_conv_loop(input_node, m, cfg, desc = 'encode', bypass_nodes = None, reuse_weights = False, batch_normalize = False, no_nonlinearity_end = False, do_print=True, return_bypass=False, sub_bypass = None, use_3d=False):
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
                                        raise ValueError('Bypass is dict but no sub_bypass specified')
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



                        with tf.contrib.framework.arg_scope([m.conv, m.conv3d], init='xavier', stddev=.01, bias=0, batch_normalize = norm_it):
                            cfs = cfg[desc][i]['conv']['filter_size']
                            cfs0 = cfs
                            nf = cfg[desc][i]['conv']['num_filters']
                            cs = cfg[desc][i]['conv']['stride']
                            if do_print:
                                print('conv in', m.output)
                            if no_nonlinearity_end and i == encode_depth:
                                if use_3d:
                                    m.conv3d(nf, cfs, cs, activation = None)
                                else:
                                    m.conv(nf, cfs, cs, activation = None)
                            else:
                                my_activation = cfg[desc][i].get('nonlinearity')
                                if my_activation is None:
                                        my_activation = 'relu'
                                else:
                                        print('NONLIN: ' + my_activation)
                                if use_3d:
                                    m.conv3d(nf, cfs, cs, activation = my_activation)
                                else:
                                    m.conv(nf, cfs, cs, activation = my_activation)
                            if do_print:
                                print('conv out', m.output)
        #TODO add print function
                        pool = cfg[desc][i].get('pool')
                        if pool:
                            pfs = pool['size']
                            ps = pool['stride']
                            if use_3d:
                                m.pool3d(pfs, ps)
                            else:
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


def just_actions_bench(inputs, cfg = None, num_classes = None, time_seen = None, normalization_method = None, stats_file = None, add_gaussians = True, image_height = None, image_width = None, **kwargs):
        base_net = fp_base.ShortLongFuturePredictionBase(inputs, store_jerk = True, normalization_method = normalization_method,
                                        time_seen = time_seen, stats_file = stats_file, add_gaussians = add_gaussians, img_height = image_height,
                                        img_width = image_width)
        m = ConvNetwithBypasses(**kwargs)
        act_node = base_net.inputs['actions_no_pos']
        act_shape = act_node.get_shape().as_list()
        batch_size = act_shape[0]
        m.output = act_node
        act_node = m.reshape([np.prod(act_shape[1:])])
        pred = hidden_loop_with_bypasses(m.output, m, cfg, reuse_weights = False, train = kwargs['train'])
        pred_shape = pred.get_shape().as_list()
        if num_classes is not None:
                pred_shape.append(int(num_classes))
                pred_shape[1] = int(pred_shape[1] / num_classes)
                pred = tf.reshape(pred, pred_shape)
        retval = {'pred' : pred}
        retval.update(base_net.inputs)
        return retval, m.params





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

def mom_model_step2(inputs, cfg = None, time_seen = None, normalization_method = None,
        stats_file = None, obj_pic_dims = None, scale_down_height = None,
        scale_down_width = None, add_depth_gaussian = False, add_gaussians = False,
        use_segmentation = False, use_vel = False, include_pose = False, 
        use_only_t1 = True, do_reconstruction = False,
        num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
    print('------NETWORK START-----')
    with tf.device('/gpu:%d' % gpu_id):
        # rescale inputs to be divisible by 8
        rinputs = {}
        for k in inputs:
            if k in ['depths', 'objects', 'vels', 'accs', 'jerks',
                    'vels_curr', 'accs_curr', 'actions_map', 'segmentation_map']:
                rinputs[k] = tf.pad(inputs[k],
                        [[0,0], [0,0], [0,0], [3,3], [0,0]], "CONSTANT")
                # RESIZING IMAGES
                rinputs[k] = tf.unstack(rinputs[k], axis=1)
                for i, _ in enumerate(rinputs[k]):
                    rinputs[k][i] = tf.image.resize_images(rinputs[k][i], [64, 88])
                rinputs[k] = tf.stack(rinputs[k], axis=1)
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
                add_depth_gaussian = add_depth_gaussian,
                add_gaussians = add_gaussians,
                get_hacky_segmentation_map = True,
                get_actions_map = True)
        inputs = base_net.inputs

        # init network
        m = ConvNetwithBypasses(**kwargs)
        # encode per time step
        main_attributes = ['depths']
        if use_vel:
            print('Using current velocities as input')
            main_attributes.append('vels_curr_normed')
        if use_segmentation:
            print('Using segmentations as input')
            main_attributes.append('segmentation_map')
        main_input_per_time = [tf.concat([tf.cast(inputs[nm][:, t], tf.float32) \
                for nm in main_attributes], axis = 3) for t in range(time_seen)]
        if do_reconstruction:
            print('Doing reconstruction only!')
            main_input_per_time = []
            for t in range(time_seen):
                inp_t = tf.concat([inputs['depths'][:,t], inputs['vels_normed'][:,t+1]],
                        axis = 3)
                main_input_per_time.append(inp_t)

        # initial bypass
        bypass_nodes = [inputs['depths'][:, 1], inputs['vels_curr_normed'][:, 1]]
        if do_reconstruction:
            bypass_nodes = [inputs['depths'][:, 1], inputs['vels_normed'][:, 2]]

        # conditioning
        if 'use_cond' in cfg:
            use_cond = cfg['use_cond']
        else:
            use_cond = False
        if use_cond:
            print('Using ACTION CONDITIONING')
            cond_attributes = ['actions_map']
            inputs['actions_map'] = tf.reduce_sum(inputs['actions_map'], axis=-1)
            if 'cond_scale_factor' in cfg:
                scale_factor = cfg['cond_scale_factor']
            else:
                scale_factor = 1
            for att in cond_attributes:
                shape = inputs[att].get_shape().as_list()
                inputs[att] = tf.unstack(inputs[att], axis=1)
                for t, _ in enumerate(inputs[att]):
                    inputs[att][t] = tf.image.resize_images(inputs[att][t],
                            [shape[2]/scale_factor, shape[3]/scale_factor],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                inputs[att] = tf.stack(inputs[att], axis=1)
            cond_input_per_time = [tf.concat([inputs[nm][:, t] \
                    for nm in cond_attributes], axis = 3) for t in range(time_seen)]

            encoded_input_cond = []
            reuse_weights = False
            for t in range(time_seen):
                if use_only_t1 and t != 1:
                    continue
                enc, bypass_nodes = feedforward_conv_loop(
                        cond_input_per_time[t], m, cfg, desc = 'cond_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                encoded_input_cond.append(enc[-1])
                reuse_weights = True

        # main
        encoded_input_main = []
        reuse_weights = False
        for t in range(time_seen):
                if use_only_t1 and t != 1:
                    continue
                enc, bypass_nodes = feedforward_conv_loop(
                        main_input_per_time[t], m, cfg, desc = 'main_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                encoded_input_main.append(enc[-1])
                reuse_weights = True

        # concat main and cond
        if use_cond:
            reuse_weights = False
            for t in range(time_seen):
                if use_only_t1 and t != 0:
                    continue
                enc = tf.concat([encoded_input_main[t], encoded_input_cond[t]], axis=3)
                enc, bypass_nodes = feedforward_conv_loop(
                        enc, m, cfg, desc = 'encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                encoded_input_main[t] = enc[-1]
                reuse_weights = True

        # predict next moments via residuals (delta moments)
        next_moments = []
        delta_moments = []
        reuse_weights = False
        next_moment = []
        delta_moment = []
        # caluclate 1st next moments
        for t in range(time_seen-1):
            if use_only_t1:
                if t != 0:
                    continue
                enco = encoded_input_main[t]
            else:
                enco = encoded_input_main[t+1]
            dm, bypass_nodes = feedforward_conv_loop(
                    enco, m, cfg, desc = 'delta_moments_encode',
                    bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                    batch_normalize = False, no_nonlinearity_end = False,
                    do_print=(not reuse_weights), return_bypass = True)
            if cfg['combine_delta'] == 'plus':
                print('Using PLUS')
                nm = enco + dm[-1]
            elif cfg['combine_delta'] == 'concat':
                print('Using CONCAT')
                nm = tf.concat([enco, dm[-1]], axis=3)
                nm, bypass_nodes = feedforward_conv_loop(
                    nm, m, cfg, desc = 'combine_delta_encode',
                    bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                    batch_normalize = False, no_nonlinearity_end = False,
                    do_print=(not reuse_weights), return_bypass = True)
                nm = nm[-1]
            else:
                raise KeyError('Unknown combine_delta')
            reuse_weights = True
            delta_moment.append(dm[-1])
            next_moment.append(nm)
        next_moments.append(next_moment)
        delta_moments.append(delta_moment)
        num_deconv = cfg.get('deconv_depth')
        reuse_weights = False
        if num_deconv:
            for moment in next_moments:
                for t, _ in enumerate(moment):
                    enc, bypass_nodes = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True)
                    moment[t] = enc[-1]
                    reuse_weights = True
            for moment in delta_moments:
                for t, _ in enumerate(moment):
                    enc, bypass_nodes = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True)
                    moment[t] = enc[-1]
                    reuse_weights = True
        retval = {
                'pred_next_vel_1': next_moments[0][0],
                #'pred_next_vel_2': next_moments[0][1],
                'bypasses': bypass_nodes,
                'delta_moments': delta_moments,
                'next_moments': next_moments
                }
        retval.update(base_net.inputs)
        print('------NETWORK END-----')
        print('------BYPASSES-------')
        for i, bypass_node in enumerate(bypass_nodes):
            print(i, bypass_node)
        print(len(bypass_nodes))
        return retval, m.params

def create_meshgrid(z_t):
    with tf.variable_scope('meshgrid'):
        shape = z_t.get_shape().as_list()
        assert len(shape) == 4 and shape[3] == 1,\
                ('Input has to be of shape [batch_size, height, width, 1]')
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        x_t = tf.tile(tf.expand_dims(\
                tf.matmul(tf.ones(shape=tf.stack([height, 1])),\
                tf.transpose(tf.expand_dims(\
                tf.linspace(-1.0, 1.0, width), 1), [1, 0])),\
                axis=0), [batch_size, 1, 1])
        y_t = tf.tile(tf.expand_dims(\
                tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),\
                tf.ones(shape=tf.stack([1, width]))),\
                axis=0), [batch_size, 1, 1])
        z_t = tf.squeeze(z_t)
        ones = tf.ones_like(x_t)
       
        grid = tf.stack([x_t, y_t, z_t, ones], axis=-1)
        return grid

def apply_projection(inputs, P):
    assert P.get_shape().as_list() == [4,4], 'P must be of shape [4,4]'
    inputs_shape = inputs.get_shape().as_list()
    assert len(inputs_shape) == 4 and inputs_shape[3] == 1,\
            ('Input has to be of shape [batch_size, height, width, 1]')
    with tf.variable_scope('projection'):
        inputs = create_meshgrid(inputs)
        inputs_shape = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1, 4])
        inputs = tf.matmul(inputs, P)
        inputs = tf.reshape(inputs, inputs_shape)
        return inputs[:,:,:,0:3]

def reverse_projection(inputs, P):
    assert P.get_shape().as_list() == [4,4], 'P must be of shape [4,4]'
    inputs_shape = inputs.get_shape().as_list()
    assert len(inputs_shape) == 4 and inputs_shape[3] == 3,\
            ('Input has to be of shape [batch_size, height, width, 3]')
    with tf.variable_scope('reverse_projection'):
        P_inv = tf.matrix_inverse(P)
        inputs = tf.concat([inputs, tf.ones_like(inputs[:,:,:,0:1])], axis=-1)
        inputs_shape = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1, 4])
        inputs = tf.matmul(inputs, P_inv)
        inputs = tf.reshape(inputs, inputs_shape)
        inputs = tf.reduce_max(inputs, axis=3, keep_dims=True)
        return tf.maximum(0, inputs)

def particle_model(inputs, cfg = None, time_seen = None, normalization_method = None,
        stats_file = None, num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
    print('------NETWORK START-----')
    with tf.device('/gpu:%d' % gpu_id):

        P = np.array([
            [1.30413,    0.00000,    0.00000,   0.00000],
            [0.00000,    1.73205,    0.00000,   0.00000],
            [0.00000,    0.00000,   -1.00060,  -0.60018],
            [0.00000,    0.00000,   -1.00000,   0.00000]]).transpose().astype(np.float32)

        P_inv = np.array([
            [0.76679,    0.00000,    0.00000,   0.00000],
            [0.00000,    0.57735,    0.00000,   0.00000],
            [0.00000,    0.00000,    0.00000,  -1.000005],
            [0.00000,    0.00000,   -1.66617,   1.66717]]).transpose().astype(np.float32)

        # rescale inputs to be divisible by 8
        rinputs = {}
        for k in inputs:
            pd = 3 #padding 3
            nh = 64 #new height 64
            nw = 88 #new width 88
            nd = 64 #new depth 64
            if k in ['depths', 'objects', 'vels', 'accs', 'jerks',
                    'vels_curr', 'accs_curr', 'actions_map', 'segmentation_map']:
                rinputs[k] = tf.pad(inputs[k],
                        [[0,0], [0,0], [0,0], [pd,pd], [0,0]], "CONSTANT")
                # RESIZING IMAGES
                rinputs[k] = tf.unstack(rinputs[k], axis=1)
                for i, _ in enumerate(rinputs[k]):
                    rinputs[k][i] = tf.image.resize_images(rinputs[k][i], [nh, nw])
                rinputs[k] = tf.stack(rinputs[k], axis=1)
            else:
                rinputs[k] = inputs[k]

       # preprocess input data
        batch_size, time_seen, height, width = \
                rinputs['depths'].get_shape().as_list()[:4]
        assert time_seen == 3, 'Wrong input data time'
        time_seen -= 1
        base_net = fp_base.ShortLongFuturePredictionBase(
                rinputs, store_jerk = False,
                normalization_method = normalization_method,
                time_seen = time_seen, stats_file = stats_file,
                scale_down_height = None,
                scale_down_width = None,
                add_depth_gaussian = False,
                add_gaussians = False,
                get_hacky_segmentation_map = True, #TODO HACKY only use six dataset!!!
                get_actions_map = True)
        inputs = base_net.inputs

        # decode depth images
        depths = tf.cast(inputs['depths_raw'], tf.float32)
        depths = -(depths[:,0:2,:,:,0:1] * 256 + depths[:,0:2,:,:,1:2] + \
                depths[:,0:2,:,:,2:3] / 256.0) / 1000.0
        depths = (P[2,2] * depths + P[3,2]) / (P[2,3] * depths)
        depths = tf.unstack(depths, axis=1)
        assert len(depths) == 2, 'Wrong time seen input length'
        points = []
        for i, depth in enumerate(depths):
            depth = create_meshgrid(depth)
            depth_shape = tf.shape(depth)
            depth = tf.matmul(tf.reshape(depth, [-1,4]), P_inv)
            depth = tf.reshape(depth, depth_shape)
            depth = tf.concat([depth[:,:,:,0:3] / depth[:,:,:,3:], depth[:,:,:,3:]], -1)
            points.append(depth)

        # decode velocities
        next_vels = tf.cast(inputs['vels'][:,2], tf.float32)
        next_vels = (next_vels - 127) / 255 / 0.5 * 2
        next_vels = tf.concat([next_vels[:,:,:,0:1],
            -next_vels[:,:,:,1:2],
            next_vels[:,:,:,2:3],
            tf.zeros(next_vels.get_shape().as_list()[:4])], axis=3)

        curr_vels = tf.cast(inputs['vels_curr'][:,1], tf.float32)
        curr_vels = (curr_vels - 127) / 255 / 0.5 * 2
        curr_vels = tf.concat([curr_vels[:,:,:,0:1],
            -curr_vels[:,:,:,1:2],
            curr_vels[:,:,:,2:3],
            tf.zeros(curr_vels.get_shape().as_list()[:4])], axis=3)

        seg_maps = tf.reduce_sum(inputs['segmentation_map'], axis=-1, keep_dims=True)
        act_maps = tf.reduce_sum(inputs['actions_map'], axis=-1, keep_dims=False)

        # assemble current state (only points[-1] is important here)
        # next_vels is also conatenated here but later removed for ground truth grid
        # TODO later do not concatenate curr_vels and seg_maps but learn to estimate them
        for i, pts in enumerate(points):
            state = [pts, curr_vels, seg_maps[:,i], act_maps[:,i], next_vels]
            points[i] = tf.concat(state, axis=-1)

        # flatten 2d points matrix, with 4d homogenous coordinates and features
        for i, pts in enumerate(points):
            shape = pts.get_shape().as_list()
            points[i] = tf.reshape(pts, [batch_size, shape[1]*shape[2], -1])
        # transform to occupancy grid
        # CAREFUL! GRID IS SWAPPED IN X AND Y COMPARED TO IMAGE!
        discretization_shape = np.array([nw, nh, nd]).astype(np.int32) # 170, 128, 128
        grids = []
        indices = []
        for i, pts in enumerate(points):
            assert len(pts.get_shape().as_list()) == 3
            batch_size, n_pts, feature_dim = tf.unstack(tf.shape(pts))[0:3]
            vals = tf.reshape(pts, [-1, feature_dim])
            # transform to projection space (all coordinates between -1 and 1)
            # but depth is 1/depth here! 
            # -> use continous camera space coordinates as channels
            # but projection space for occupancy grid: inverse depth relation???
            idxs = tf.reshape(pts[:,:,0:4], [-1, 4])
            idxs = tf.matmul(idxs, P)
            # TODO clip or not? might be that we just want to push outside camera points
            # to edges as storage for rotating them back later
            idxs = tf.clip_by_value(idxs[:,0:3] / idxs[:,3:], -1.0, 1.0)
            idxs = tf.cast(tf.round(
                (idxs + 1) / 2 * (discretization_shape - 1)), tf.int32)
            batch_idxs = tf.reshape(tf.tile(tf.expand_dims(
                tf.range(batch_size), 1), [1, n_pts]), [-1, 1])
            idxs = tf.concat([batch_idxs, idxs], 1)
            grid_shape = tf.stack([batch_size] \
                    + tf.unstack(discretization_shape) + [feature_dim], 0) 
            grid = tf.scatter_nd(idxs, vals, grid_shape, name='grid_'+str(i))
            grids.append(grid)
            indices.append(idxs)

        # init network
        m = ConvNetwithBypasses(**kwargs)

        # encode per time step
        # action as an external effect will be concatenated on again later
        #TODO later when vel estimated add grids[0]
        state_grid, action_grid, next_vel_grid = tf.split(grids[1], [9, 6, 4], axis=4)
        main_input_per_time = [state_grid] # remove action
                
        # initial bypass, state and external actions
        bypass_nodes = [[state_grid, action_grid]]

        # main
        encoded_input_main = []
        reuse_weights = False
        for t in range(time_seen-1): # TODO -1 as first input skipped as velocity input
                enc, bypass_nodes[t] = feedforward_conv_loop(
                        main_input_per_time[t], m, cfg, desc = 'main_encode',
                        bypass_nodes = bypass_nodes[t], reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True, use_3d=True)
                encoded_input_main.append(enc[-1])
                reuse_weights = True
        pred_vel = encoded_input_main[0]

        pred_vel_flat = tf.reshape(tf.gather_nd(pred_vel, indices[0]), \
                [batch_size, nw, nh, 3])
        pred_vel_flat = tf.concat([
            pred_vel_flat[:,:,:,0:1],
            -pred_vel_flat[:,:,:,1:2],
            pred_vel_flat[:,:,:,2:3]], -1)
        pred_vel_flat = tf.clip_by_value(pred_vel_flat, -2.0, 2.0) / 2.0 * 0.5*255 + 127
        pred_vel_flat = tf.cast(pred_vel_flat, tf.uint8)

        retval = {
                'pred_vel_flat': pred_vel_flat,
                'pred_vel': pred_vel,
                'state': state_grid,
                'next_vel': next_vel_grid,
                'bypasses': bypass_nodes,
                }
        retval.update(base_net.inputs)
        print('------NETWORK END-----')
        print('------BYPASSES-------')
        for i, node in enumerate(bypass_nodes[0]):
            print(i, bypass_nodes[0][i])
        return retval, m.params

def mom_complete(inputs, cfg = None, time_seen = None, normalization_method = None,
        stats_file = None, obj_pic_dims = None, scale_down_height = None,
        scale_down_width = None, add_depth_gaussian = False, add_gaussians = False,
        include_pose = False, store_jerk = True, use_projection = False, 
        num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
    print('------NETWORK START-----')
    with tf.device('/gpu:%d' % gpu_id):
        # rescale inputs to be divisible by 8
        rinputs = {}
        for k in inputs:
            if k in ['depths', 'objects', 'vels', 'accs', 'jerks',
                    'vels_curr', 'accs_curr', 'actions_map', 'segmentation_map']:
                rinputs[k] = tf.pad(inputs[k],
                        [[0,0], [0,0], [0,0], [3,3], [0,0]], "CONSTANT")
                # RESIZING IMAGES
                rinputs[k] = tf.unstack(rinputs[k], axis=1)
                for i, _ in enumerate(rinputs[k]):
                    rinputs[k][i] = tf.image.resize_images(rinputs[k][i], [64, 88])
                rinputs[k] = tf.stack(rinputs[k], axis=1)
            else:
                rinputs[k] = inputs[k]
       # preprocess input data
        batch_size, time_seen, height, width = \
                rinputs['depths'].get_shape().as_list()[:4]
        time_seen -= 1
        long_len = rinputs['object_data'].get_shape().as_list()[1]
        base_net = fp_base.ShortLongFuturePredictionBase(
                rinputs, store_jerk = store_jerk,
                normalization_method = normalization_method,
                time_seen = time_seen, stats_file = stats_file,
                scale_down_height = scale_down_height,
                scale_down_width = scale_down_width,
                add_depth_gaussian = add_depth_gaussian,
                add_gaussians = add_gaussians,
                get_hacky_segmentation_map = True,
                get_actions_map = True)
        inputs = base_net.inputs

        # init network
        m = ConvNetwithBypasses(**kwargs)

        # encode per time step
        main_attributes = ['depths']
        main_input_per_time = [tf.concat([tf.cast(inputs[nm][:, t], tf.float32) \
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
        cond_input_per_time = [tf.concat([inputs[nm][:, t] \
                for nm in cond_attributes], axis = 3) for t in range(time_seen)]

        encoded_input_cond = []
        reuse_weights = False
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
                    enc = tf.concat([sm[t+1], sm[t]], axis=3)
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
                enc = tf.concat([moment[t], 
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
                    nm = tf.concat([current[t], dm[-1]], axis=3)
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
		enc = tf.concat([moment[t], encoded_input_main[t+i]], axis=3)
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
        retval.update(base_net.inputs)
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

def mom_model(inputs, cfg = None, time_seen = None, normalization_method = None,
        stats_file = None, obj_pic_dims = None, scale_down_height = None,
        scale_down_width = None, add_depth_gaussian = False, add_gaussians = False, 
        include_pose = False,
        num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
    print('------NETWORK START-----')
    with tf.device('/gpu:%d' % gpu_id):
        # rescale inputs to be divisible by 8
        rinputs = {}
        for k in inputs:
            if k in ['depths', 'objects', 'vels', 'accs', 'jerks', 
                    'vels_curr', 'accs_curr', 'actions_map']:
                rinputs[k] = tf.pad(inputs[k],
                        [[0,0], [0,0], [0,0], [3,3], [0,0]], "CONSTANT")
                # RESIZING IMAGES
                rinputs[k] = tf.unstack(rinputs[k], axis=1)
                for i, _ in enumerate(rinputs[k]):
                    rinputs[k][i] = tf.image.resize_images(rinputs[k][i], [64, 88])
                rinputs[k] = tf.stack(rinputs[k], axis=1)
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
                add_depth_gaussian = add_depth_gaussian,
                add_gaussians = add_gaussians,
                get_actions_map = True)
        inputs = base_net.inputs

        # init network
        m = ConvNetwithBypasses(**kwargs)
        # encode per time step
        main_attributes = ['depths']
        main_input_per_time = [tf.concat([inputs[nm][:, t] \
                for nm in main_attributes], axis = 3) for t in range(time_seen)]

        # initial bypass
        bypass_nodes = [inputs['depths'][:, time_seen-1]]

        # conditioning
        if 'use_cond' in cfg:
            use_cond = cfg['use_cond']
        else:
            use_cond = False
        if use_cond:
            print('Using CONDITIONING')
            cond_attributes = ['actions_map']
            inputs['actions_map'] = tf.reduce_sum(inputs['actions_map'], axis=-1)
            if 'cond_scale_factor' in cfg:
                scale_factor = cfg['cond_scale_factor']
            else:
                scale_factor = 1
            for att in cond_attributes:
                shape = inputs[att].get_shape().as_list()
                inputs[att] = tf.unstack(inputs[att], axis=1)
                for t, _ in enumerate(inputs[att]):
                    inputs[att][t] = tf.image.resize_images(inputs[att][t],
                            [shape[2]/scale_factor, shape[3]/scale_factor],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                inputs[att] = tf.stack(inputs[att], axis=1)
            cond_input_per_time = [tf.concat([inputs[nm][:, t] \
                    for nm in cond_attributes], axis = 3) for t in range(time_seen)]

            encoded_input_cond = []
            reuse_weights = False
            for t in range(time_seen):
                enc, bypass_nodes = feedforward_conv_loop(
                        cond_input_per_time[t], m, cfg, desc = 'cond_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                encoded_input_cond.append(enc[-1])
                reuse_weights = True

        # main
        encoded_input_main = []
        reuse_weights = False
        for t in range(time_seen):
                enc, bypass_nodes = feedforward_conv_loop(
                        main_input_per_time[t], m, cfg, desc = 'main_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                encoded_input_main.append(enc[-1])
                reuse_weights = True

        # concat main and cond
        if use_cond:
            reuse_weights = False
            for t in range(time_seen):
                enc = tf.concat([encoded_input_main[t], encoded_input_cond[t]], axis=3)
            	enc, bypass_nodes = feedforward_conv_loop(
                        enc, m, cfg, desc = 'encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                encoded_input_main[t] = enc[-1]
                reuse_weights = True

        # calculate 1st moments
        moments = []
        first_moments = []
        reuse_weights = False
        for t in range(time_seen-1):
            if cfg['combine_moments'] == 'minus':
                print('Using MINUS')
                enc = encoded_input_main[t+1] - encoded_input_main[t]
            elif cfg['combine_moments'] == 'concat':
                print('Using CONCAT')
                enc = tf.concat([encoded_input_main[t+1], encoded_input_main[t]], axis=3)
                enc, bypass_nodes = feedforward_conv_loop(
                        enc, m, cfg, desc = 'combine_moments_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                enc = enc[-1]
            enc, bypass_nodes = feedforward_conv_loop(
                    enc, m, cfg, desc = 'moments_encode',
                    bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                    batch_normalize = False, no_nonlinearity_end = False,
                    do_print=(not reuse_weights), return_bypass = True)
            first_moments.append(enc[-1])
            reuse_weights = True
        moments.append(first_moments)

        # calculate 2nd moments while reusing weights
        second_moments = []
        reuse_weights = True
        for t in range(time_seen-2):
            if cfg['combine_moments'] == 'minus':
                print('Using MINUS')
                enc = moments[0][t+1] - moments[0][t]
            elif cfg['combine_moments'] == 'concat':
                print('Using CONCAT')
                enc = tf.concat([moments[0][t+1], moments[0][t]], axis=3)
                enc, bypass_nodes = feedforward_conv_loop(
                        enc, m, cfg, desc = 'combine_moments_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                enc = enc[-1]
            else:
                raise KeyError('Unknown combine_moments')
            enc, bypass_nodes = feedforward_conv_loop(
                    enc, m, cfg, desc = 'moments_encode',
                    bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                    batch_normalize = False, no_nonlinearity_end = False,
                    do_print=(not reuse_weights), return_bypass = True)
            second_moments.append(enc[-1])
            reuse_weights = True
        moments.append(second_moments)

        # predict next moments via residuals (delta moments)
        next_moments = []
        delta_moments = []
        reuse_weights = False
        for moment in moments:
            next_moment = []
            delta_moment = []
            for t, _ in enumerate(moment):
                dm, bypass_nodes = feedforward_conv_loop(
                        moment[t], m, cfg, desc = 'delta_moments_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                if cfg['combine_delta'] == 'plus':
                    print('Using PLUS')
                    nm = moment[t] + dm[-1]
                elif cfg['combine_delta'] == 'concat':
                    print('Using CONCAT')
                    nm = tf.concat([moment[t], dm[-1]], axis=3)
                    nm, bypass_nodes = feedforward_conv_loop(
                        nm, m, cfg, desc = 'combine_delta_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=(not reuse_weights), return_bypass = True)
                    nm = nm[-1]
                else:
                    raise KeyError('Unknown combine_delta')
                reuse_weights = True
                delta_moment.append(dm[-1])
                next_moment.append(nm)
            next_moments.append(next_moment)
            delta_moments.append(delta_moment)

        # encode zero delta moments (pos -> vel)
        delta_moment = []
        reuse_weights = True
        for t, _ in enumerate(encoded_input_main):
            dm, bypass_nodes = feedforward_conv_loop(
		    encoded_input_main[t], m, cfg, desc = 'delta_moments_encode',
		    bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
		    batch_normalize = False, no_nonlinearity_end = False,
		    do_print=(not reuse_weights), return_bypass = True)
            reuse_weights = True
            delta_moment.append(dm[-1])
        delta_moments.append(delta_moment)

        num_deconv = cfg.get('deconv_depth')
        reuse_weights = False
        if num_deconv:
            for moment in moments:
                for t, _ in enumerate(moment):
                    enc, bypass_nodes = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True)
                    moment[t] = enc[-1]
                    reuse_weights = True
            for moment in next_moments:
                for t, _ in enumerate(moment):
                    enc, bypass_nodes = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True)
                    moment[t] = enc[-1]
                    reuse_weights = True
            for moment in delta_moments:
                for t, _ in enumerate(moment):
                    enc, bypass_nodes = deconv_loop(
                            moment[t], m, cfg, desc='deconv',
                            bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                            batch_normalize = False, no_nonlinearity_end = False,
                            do_print = True, return_bypass = True)
                    moment[t] = enc[-1]
                    reuse_weights = True
        retval = {
                'pred': delta_moments[1][0],
                'pred_vel_1': moments[0][0],
                'pred_next_vel_1': next_moments[0][0],
                'pred_next_vel_2': next_moments[0][1],
                'bypasses': bypass_nodes,
                'moments': moments,
                'delta_moments': delta_moments,
                'next_moments': next_moments
                }
        retval.update(base_net.inputs)
        #print('------BYPASSES-------')
        #for bypass_node in bypass_nodes:
        #    print(bypass_node)
        #print(len(bypass_nodes))
        print('------NETWORK END-----')
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
            if k in ['depths', 'objects', 'jerks']:
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

def map_jerk_action_model(inputs, cfg = None, time_seen = None, 
        normalization_method = None, 
        stats_file = None, obj_pic_dims = None, scale_down_height = None, 
        scale_down_width = None, add_depth_gaussian = False, include_pose = False, 
        num_classes = None, keep_prob = None, gpu_id = 0, **kwargs):
    print('------NETWORK START-----')
    with tf.device('/gpu:%d' % gpu_id):
        # rescale inputs to be divisible by 8
        rinputs = {}
        for k in inputs:
            if k in ['depths', 'objects', 'jerks', 'actions_map']:
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
        main_attributes = ['depths']
        main_input_per_time = [tf.concat([inputs[nm][:, t] \
                for nm in main_attributes], axis = 3) for t in range(time_seen)]

        cond_attributes = ['actions_map']
        if 'cond_scale_factor' in cfg:
            scale_factor = cfg['cond_scale_factor']
        else:
            scale_factor = 1
        for att in cond_attributes:
            shape = inputs[att].get_shape().as_list()
            inputs[att] = tf.unstack(inputs[att], axis=1)
            for t, _ in enumerate(inputs[att]):
                inputs[att][t] = tf.image.resize_images(inputs[att][t],
                        [shape[2]/scale_factor, shape[3]/scale_factor],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            inputs[att] = tf.stack(inputs[att], axis=1)
        cond_input_per_time = [tf.concat([inputs[nm][:, t] \
                for nm in cond_attributes], axis = 3) for t in range(time_seen)]

        encoded_input_cond = []
        bypass_nodes = [inputs['depths'][:, time_seen-1]]
        reuse_weights = False
        do_print = True
        for t in range(time_seen):
                # size 1
                enc, bypass_nodes = feedforward_conv_loop(
                        cond_input_per_time[t], m, cfg, desc = 'cond_encode', 
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights, 
                        batch_normalize = False, no_nonlinearity_end = False, 
                        do_print=do_print, return_bypass = True)
                encoded_input_cond.append(enc[-1])
                reuse_weights = True
        	do_print = False
        
        # size 2
        encoded_input_main = []
        reuse_weights = False
        do_print = True
        for t in range(time_seen):
                enc, bypass_nodes = feedforward_conv_loop(
                        main_input_per_time[t], m, cfg, desc = 'main_encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
                        batch_normalize = False, no_nonlinearity_end = False,
                        do_print=do_print, return_bypass = True)
                encoded_input_main.append(enc[-1])
                do_print = False
                reuse_weights = True
        # concat and encode per time
        encoded_input = []
        reuse_weights = False
        do_print = True
        for t in range(time_seen):
            enc = tf.concat([encoded_input_main[t], encoded_input_cond[t]], axis=3)
            enc, bypass_nodes = feedforward_conv_loop(
                        enc, m, cfg, desc = 'encode',
                        bypass_nodes = bypass_nodes, reuse_weights = reuse_weights,
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
        print('------BYPASSES-------')
        for bypass_node in bypass_nodes:
            print(bypass_node)
        print(len(bypass_nodes))
        retval['bypasses'] = bypasses
        print('------NETWORK END-----')
        return retval, m.params



def stripped_down_depths_model(inputs, cfg = None, normalization_method = None, stats_file = None, num_classes = None, depth_cutoff = None, 
					include_action = True, **kwargs):
	batch_size, time_seen = inputs['depths'].get_shape().as_list()[:2]
	long_len  = inputs['object_data'].get_shape().as_list()[1]
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, store_jerk = True, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file,
				add_gaussians = False, depth_cutoff = depth_cutoff)
	inputs = base_net.inputs
	img_attributes = ['depths', 'depths2']
	input_per_time = [tf.concat([tf.expand_dims(inputs[nm][:, t], axis = 3) for nm in img_attributes], axis = 3) for t in range(time_seen)]
	m = ConvNetwithBypasses(**kwargs)

	reuse_weights = False
	encoding = []	
	for t in range(time_seen):
		encoding.append(feedforward_conv_loop(input_per_time[t], m, cfg, desc = 'encode', bypass_nodes = None, reuse_weights = reuse_weights, batch_normalize = False,
							no_nonlinearity_end = False)[-1])
		reuse_weights = True

	num_encode_together = cfg.get('encode_together_depth')
	if num_encode_together:
		print('Encoding together!')
		together_input = tf.concat(encoding, axis = 3)
		encoding = feedforward_conv_loop(together_input, m, cfg, desc = 'encode_together', bypass_nodes = [input_per_time[-1]], reuse_weights = False, batch_normalize = False,
								no_nonlinearity_end = False)[-1]

	flattened_input = [tf.reshape(encoding, [batch_size, -1])]
	if include_action:
		flattened_input.append(tf.reshape(inputs['actions_no_pos'], [batch_size, -1]))


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
	





def basic_jerk_model(inputs, cfg = None, time_seen = None, normalization_method = None, stats_file = None, obj_pic_dims = None, scale_down_height = None, scale_down_width = None, add_depth_gaussian = False, include_pose = False, num_classes = None, keep_prob = None, depths_not_normals_images = False, depth_cutoff = None, gpu_id = 0, **kwargs):
#    with tf.device('/gpu:%d' % gpu_id):
	if depths_not_normals_images:
		batch_size, time_seen = inputs['depths'].get_shape().as_list()[:2]
	else:
		batch_size, time_seen = inputs['normals'].get_shape().as_list()[:2]
	long_len = inputs['object_data'].get_shape().as_list()[1]
	base_net = fp_base.ShortLongFuturePredictionBase(inputs, store_jerk = True, normalization_method = normalization_method, time_seen = time_seen, stats_file = stats_file, scale_down_height = scale_down_height, scale_down_width = scale_down_width, add_depth_gaussian = add_depth_gaussian, depth_cutoff = depth_cutoff)

	inputs = base_net.inputs

	if depths_not_normals_images:
		size_1_attributes = ['depths', 'depths2']
		size_1_input_per_time = [tf.concat([tf.expand_dims(inputs[nm][:, t], axis = 3)  for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
	else:
		size_1_attributes = ['normals', 'normals2', 'images']
		size_1_input_per_time = [tf.concat([inputs[nm][:, t] for nm in size_1_attributes], axis = 3) for t in range(time_seen)]
	size_2_attributes = ['object_data_seen', 'actions_seen']	
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



def correlation_jerk_loss(outputs, l2_coef = 1., corr_coef = 1.):
        pred = outputs['pred']
        tv = outputs['jerk']
        n_entries = np.prod(tv.get_shape().as_list())
        return l2_coef * tf.nn.l2_loss(pred - tv) / n_entries - corr_coef * (correlation(pred, tv) + 1)



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

def int_shape(x):
    return list(map(int, x.get_shape()))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))

def discretized_mix_logistic_loss(outputs, gpu_id=0, buckets = 255.0, 
        sum_all=True, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
        x = (outputs['jerk_map'] + 1) / 2 * buckets
        l = outputs['pred']
        """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
        xs = int_shape(
            x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
        ls = int_shape(l)  # predicted distribution, e.g. (B,32,32,100)
        # here and below: unpacking the params of the mixture of logistics
        nr_mix = int(ls[-1] / 10)
        logit_probs = l[:, :, :, :nr_mix]
        l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
        means = l[:, :, :, :, :nr_mix]
        log_scales = tf.maximum(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
        coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
        # here and below: getting the means and adjusting them based on preceding
        # sub-pixels
        x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
        m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                        * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
        m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                        coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
        means = tf.concat([tf.reshape(means[:, :, :, 0, :], [
                          xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3)
        centered_x = x - means
        inv_stdv = tf.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 0.5)#1. / buckets)
        cdf_plus = tf.nn.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 0.5)#1. / buckets)
        cdf_min = tf.nn.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -tf.nn.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

        # now select the right output: left edge case, right edge case, normal
        # case, extremely low prob case (doesn't actually happen for us)

        # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
        # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

        # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
        # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
        # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
        # if the probability on a sub-pixel is below 1e-5, we use an approximation
        # based on the assumption that the log-density is constant in the bin of
        # the observed sub-pixel value
        
        #log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(buckets / 2))))
        log_probs = tf.log(tf.maximum(cdf_delta, 1e-12))

        log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
        if sum_all:
            return [-tf.reduce_mean(log_sum_exp(log_probs))]
        else:
            return [-tf.reduce_mean(log_sum_exp(log_probs), [1, 2])]

def particle_loss(outputs, gpu_id, **kwargs):
    state = outputs['state']
    group = state[:,:,:,:,8:9]
    positions = state[:,:,:,:,0:3]
    next_vel = outputs['next_vel'][:,:,:,:,0:3]
    pred_vel = outputs['pred_vel'][:,:,:,:,0:3]

    # PRESERVE DISTANCE LOSS
    # construct the pairwise distance calculation kernels
    # pairwise distance kernel: 
    # depth height, width, in (x,y,z), out (right, bottom, front) distance
    ks = 5
    dim = 3
    k3d = np.zeros([ks,ks,ks,dim,(ks*ks*ks-1)*dim])
    km = ks / 2 
    # set x,y,z center to 1 and boundary -1
    m = 0
    for i in range(ks):
        for j in range(ks):
            for k in range(ks):
                for l in range(dim):
                    if not(i == j == k == km):
                        k3d[i,j,k,l,m] = -1
                        k3d[km,km,km,l,m] = 1
                        m += 1

    # mask kernel: to only use distance between particles of the same group
    # select the appropriate channel 0 for particle label
    k3m = np.zeros([ks,ks,ks,1,ks*ks*ks-1])
    m = 0
    for i in range(ks):
        for j in range(ks):
            for k in range(ks):
                if not(i == j == k == km):
                    k3m[i,j,k,0,m] = 1
                    k3m[km,km,km,0,m] = 1
                    m += 1

    # determine active relations
    relation = tf.nn.conv3d(group, k3m, [1,1,1,1,1], "SAME")
    relation_same = tf.cast(
            tf.logical_and(tf.equal(relation, 2 * group), tf.not_equal(relation, 0)), 
            tf.float32)
    # determine distance between neighboring particles at time t
    positions = [positions, positions + pred_vel]
    distances = []
    for i, pos in enumerate(positions):
        distance = tf.nn.conv3d(pos, k3d, [1,1,1,1,1], "SAME")
        distance *= distance
        distance = tf.stack([tf.reduce_sum(dim, axis=-1) for dim in \
            tf.split(distance, k3m.shape[4], axis=4)])
        distances.append(distance)
    preserve_distance_loss = tf.reduce_sum((distances[1] - distances[0]) ** 2 \
            * relation_same) / 2

    # CONSERVATION OF MASS = MINIMUM DISTANCE BETWEEN PARTICLES HAS TO BE KEPT
    # with mask to enforce only between solid particles and not between empty particles
    min_distance = 0.2
    relation_solid = tf.cast(
            tf.logical_and(tf.greater(relation, group), tf.not_equal(group, 0)),
            tf.float32)
    mass_conservation_loss = tf.reduce_sum( \
            tf.nn.relu(-distances[1]+min_distance) * relation_solid)

    # MSE VELOCITY LOSS
    mse_velocity_loss = tf.nn.l2_loss(pred_vel - next_vel)
    
    # MEAN OF BOTH LOSSES
    loss = tf.reduce_mean(tf.stack([mse_velocity_loss, preserve_distance_loss]))
    
    return [loss]

def softmax_cross_entropy_loss_binary_jerk(outputs, gpu_id, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
	labels = tf.cast(tf.not_equal(
                tf.norm(outputs['jerk_map'], ord='euclidean', axis=3), 0), tf.int32)
	shape = outputs['pred'].get_shape().as_list()
	assert shape[3] == 2
	logits = outputs['pred']

        undersample = False
        if undersample:
            thres = 0.5412
            mask = tf.norm(outputs['jerk_all'], ord='euclidean', axis=2)
            mask = tf.cast(tf.logical_or(tf.greater(mask[:,0], thres),
                tf.greater(mask[:,1], thres)), tf.float32)
            mask = tf.reshape(mask, [mask.get_shape().as_list()[0], 1, 1, 1])
        else:
            mask = 1

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits) * mask)
        return [loss]

def softmax_cross_entropy_loss_depth(outputs, gpu_id = 0, eps = 0.0,
        min_value = -1.0, max_value = 1.0, num_classes=256,
        segmented_jerk=True, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
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
        # next depth losses
        logits = outputs['next_moments'][0][0]
        logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
        labels = tf.cast(outputs['depths_raw'][:,2], tf.int32)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits) * mask)
        losses.append(loss)
        assert len(losses) == 1, ('loss length: %d' % len(losses))
        losses = tf.stack(losses)
        return [tf.reduce_mean(losses)]

def softmax_cross_entropy_loss_vel_one(outputs, gpu_id = 0, eps = 0.0,
        min_value = -1.0, max_value = 1.0, num_classes=256,
        segmented_jerk=True, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
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
        labels = tf.cast(outputs['depths_raw'][:,2], tf.int32)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits) * mask)
        losses.append(loss)
        assert len(losses) == 1, ('loss length: %d' % len(losses))

        losses = tf.stack(losses)
        return [tf.reduce_mean(losses)]


def softmax_cross_entropy_loss_vel_all(outputs, gpu_id = 0, eps = 0.0,
        min_value = -1.0, max_value = 1.0, num_classes=256,
        segmented_jerk=True, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
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
        # next moment losses
        logits = outputs['next_moments'][1][0]
        logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
        labels = outputs['vels'][:,2]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits) * mask)
        losses.append(loss)
        assert len(losses) == 1, ('loss length: %d' % len(losses))

        # current moments losses
        logits = outputs['moments'][1][0]
        logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
        labels = outputs['vels_curr'][:,1]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits) * mask)
        losses.append(loss)
        assert len(losses) == 2, ('loss length: %d' % len(losses))

        # next image losses
        logits = outputs['next_images'][1][0]
        logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
        labels = tf.cast(outputs['depths_raw'][:,2], tf.int32)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits) * mask)
        losses.append(loss)
        assert len(losses) == 3, ('loss length: %d' % len(losses))

        losses = tf.stack(losses)
        return [tf.reduce_mean(losses)]

def softmax_cross_entropy_loss_vel(outputs, gpu_id = 0, eps = 0.0,
        min_value = -1.0, max_value = 1.0, num_classes=256,
        segmented_jerk=True, use_current_vel_loss=True, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
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
        # next moment losses
        logits = outputs['next_moments'][0][0]
        logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
        labels = outputs['vels'][:,2]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits) * mask)
        losses.append(loss)
        assert len(losses) == 1, ('loss length: %d' % len(losses))
        if use_current_vel_loss:
            # current moments losses
            logits = outputs['moments'][0][0]
            logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
            labels = outputs['vels_curr'][:,1]
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits) * mask)
            losses.append(loss)
            assert len(losses) == 2, ('loss length: %d' % len(losses))
        losses = tf.stack(losses)
        return [tf.reduce_mean(losses)]

def multi_moment_softmax_cross_entropy_loss_pixel_jerk(outputs, gpu_id = 0, eps = 0.0,
        min_value = -1.0, max_value = 1.0, num_classes=256, use_pos_to_vel = True, 
        segmented_jerk=True, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
        undersample = False
        if undersample:
            thres = 0.5412
            mask = tf.norm(outputs['jerk_all'], ord='euclidean', axis=2)
            mask = tf.cast(tf.logical_or(tf.greater(mask[:,0], thres),
                tf.greater(mask[:,1], thres)), tf.float32)
            mask = tf.reshape(mask, [mask.get_shape().as_list()[0], 1, 1, 1])
        else:
            mask = 1
        shape = outputs['pred'].get_shape().as_list()
        assert shape[3] / 3 == num_classes 
        
        losses = []
        # delta moment losses
        moments_labels = ['accs', 'jerks']
        for i, moment in enumerate(outputs['delta_moments'][0:2]):
            for t, _ in enumerate(moment):
                logits = moment[t]
                logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
                labels = outputs[moments_labels[i]][:,t-2+i]
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits) * mask)
                losses.append(loss)
        assert len(losses) == 3, ('loss length: %d' % len(losses))
        # next moment losses
        moments_labels = ['vels', 'accs']
        for i, moment in enumerate(outputs['next_moments'][0:2]):
            for t, _ in enumerate(moment):
                logits = moment[t]
                logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
                labels = outputs[moments_labels[i]][:,t-2+i]
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits) * mask)
                losses.append(loss)
        assert len(losses) == 6, ('loss length: %d' % len(losses))
        # current moments losses
        moments_labels = ['vels_curr', 'accs_curr']
        for i, moment in enumerate(outputs['moments'][0:2]):
            for t, _ in enumerate(moment):
                logits = moment[t]
                logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
                labels = outputs[moments_labels[i]][:,t+1+i]
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits) * mask)
                losses.append(loss)
        assert len(losses) == 9, ('loss length: %d' % len(losses))
        # delta moment losses (pos -> vel)
        if use_pos_to_vel:
            moments_labels = ['vels']
            moment = outputs['delta_moments'][2]
            for t, _ in enumerate(moment):
                logits = moment[t]
                logits = tf.reshape(logits, shape[0:3] + [3, shape[3] / 3])
                labels = outputs[moments_labels[0]][:,t+1]
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits) * mask)
                losses.append(loss)
            assert len(losses) == 12, ('loss length: %d' % len(losses))
        losses = tf.stack(losses)
        return [tf.reduce_mean(losses)]

def softmax_cross_entropy_loss_pixel_jerk(outputs, gpu_id = 0, eps = 0.0, 
        min_value = -1.0, max_value = 1.0, num_classes=256, 
        segmented_jerk=True, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
        if segmented_jerk:
            labels = tf.cast(tf.round((outputs['jerk_map'] - min_value) / \
                    (max_value - min_value) * (num_classes - 1)), tf.int32)
        else:
            labels = tf.cast(outputs['jerks'][:,-1], tf.int32)
        shape = outputs['pred'].get_shape().as_list()
        assert shape[3] / 3 == num_classes
        logits = tf.reshape(outputs['pred'], shape[0:3] + [3, shape[3] / 3])

        undersample = False
        if undersample:
            thres = 0.5412
            mask = tf.norm(outputs['jerk_all'], ord='euclidean', axis=2)
            mask = tf.cast(tf.logical_or(tf.greater(mask[:,0], thres),
                tf.greater(mask[:,1], thres)), tf.float32)
            mask = tf.reshape(mask, [mask.get_shape().as_list()[0], 1, 1, 1])
        else:
            mask = 1

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits) * mask) 
        return [loss]

def softmax_cross_entropy_loss_per_pixel(outputs, gpu_id = 0, eps = 0.01, **kwargs):
    with tf.device('/gpu:%d' % gpu_id):
        labels = tf.cast(outputs['depths_raw'][:,-1,:,:,0], tf.int32) # only predict the coarsest channel
        logits = outputs['pred']
        weight = tf.abs(outputs['jerk_map']) + eps

        undersample = False
        if undersample:
            mask = tf.norm(outputs['jerk'], ord='euclidean', axis=1)
            mask = tf.cast(tf.greater(mask, 2.473), tf.float32)
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
	print(bin_data_file)
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

def model_parallelizer(inputs, model, n_gpus, gpu_offset, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        assert n_gpus > 0, ('At least one gpu has to be used')
        if n_gpus == 1:
            return model(inputs, **kwargs)
        else:
            outputs = []
            params = []
            # split batch across GPUs
            for k in inputs:
                inputs[k] = tf.split(inputs[k], axis=0, num_or_size_splits=n_gpus)
            for i in range(n_gpus):                
                output, param = model(inputs, gpu_id = gpu_offset + i, **kwargs)
                outputs.append(output)
                params.append(param)
                tf.get_variable_scope().reuse_variables()
        outputs = dict(zip(outputs[0],zip(*[d.values() for d in outputs])))
        params = params[0]
        return [outputs, params]

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


def gen_cfg_short_jerk(num_filters_before_concat = 24, num_filters_after_concat = 34, num_filters_together = 34, encode_depth = 2, encode_size = 7, hidden_depth = 3, hidden_num_features = 250, num_classes = 1):
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
	cfg['hidden'][hidden_depth] = {'num_features' : 3 * num_classes, 'activation' : 'identity'}
	return cfg


def gen_cfg_no_explicit(num_filters_encode = [20, 20], num_filters_together = 34, encode_depth = 2, encode_size = 7, hidden_depth = 3, hidden_num_features = 250):
	cfg = {'encode_depth' : encode_depth, 'hidden_depth' : hidden_depth, 'hidden' : {}, 'encode' : {}}
	print('numbers of things')
	print(len(num_filters_encode))
	print(encode_depth)
	for i in range(1, encode_depth + 1):
		cfg['encode'][i] = {'conv' : {'filter_size' : encode_size, 'stride' : 2, 'num_filters' : num_filters_encode[i - 1]}}
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

def gen_cfg_no_explicit_alt(encode_num_filters = [4, 8, 16, 32], 
				encode_pool = [True, False, False, False], 
				encode_size = [7, 7, 7, 3],
				encode_bypasses = [None, None, None, 0],
				encode_stride = [2, 2, 2, 2],
				together_num_filters = [32],
				together_size = [1],
				together_max_pool = [False],
				together_bypasses = [0],
				hidden_num_features = [2000, 2000]):
	cfg = {'encode_depth' : len(encode_num_filters), 'encode_together_depth' : len(together_num_filters), 
				'hidden_depth' : len(hidden_num_features), 'hidden' : {}, 'encode' : {}, 'encode_together' : {}}
	for i in range(1,cfg['encode_depth'] + 1):
		cfg['encode'][i] = {'conv' : {'filter_size' : encode_size[i-1], 'stride' : encode_stride[i-1], 'num_filters' : encode_num_filters[i - 1]}, 'bypass' : encode_bypasses[i-1]}
		if encode_pool[i-1]:
			cfg['encode'][i]['pool'] = {'size' : 3, 'stride' : 2, 'type' : 'max'}
	for i in range(1, len(together_num_filters) + 1):
		cfg['encode_together'][i] = {'conv' : {'filter_size' : together_size[i - 1], 
						'stride' : 1, 'num_filters' : together_num_filters[i-1]},
						'bypass' : together_bypasses[i-1]}
		if together_max_pool[i-1]:
			cfg['encode_together'][i]['pool'] = {'size' : 3, 'stride' : 2, 'type' : 'max'}
	for i in range(1, cfg['hidden_depth']):
		cfg['hidden'][i] = {'num_features' : hidden_num_features[i-1], 'dropout' : .75}
	cfg['hidden'][cfg['hidden_depth']] = {'num_features' : 3, 'activation' : 'identity'}
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

def cfg_deep_bypass_jerk_action(n_classes):
    return {
        'cond_scale_factor': 8,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}
                    },
        },

        'main_encode_depth': 3,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 8}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16},
                    'bypass' : 0},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32},
                    'bypass' : 0},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32},
                    'bypass' : 0},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32},
                    'bypass' : 0},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_together_depth' : 12,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                3 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                4 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                5 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                6 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                7 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                8 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                9 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                10: {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                11: {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                12: {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    'bypass' : 0},
                    #, 'bypass' : 0}
        },
        'hidden_depth': 0,

        'deconv_depth': 3,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32},
                'bypass' : 0},
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16},
                'bypass' : 0},
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                'bypass' : 0},
        }
}

def cfg_no_bypass_jerk_action(n_classes):
    return {
        'cond_scale_factor': 8,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 8}
                    },
        },

        'main_encode_depth': 3,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 8}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32},
                    },
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32},
                    },
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_together_depth' : 12,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                3 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                4 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                5 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                6 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                7 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                8 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                9 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                10: {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                11: {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                12: {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 64},
                    },
                    #, 'bypass' : 0}
        },
        'hidden_depth': 0,

        'deconv_depth': 3,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 32},
                },
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 16},
                },
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                },
        }
}

def cfg_mom_complete_flat(n_classes, use_segmentation=True, 
        method='sign', nonlin='relu'):
    return {
        'use_segmentation': use_segmentation,
        # ONLY USED IF use_cond = True!!!
        'cond_scale_factor': 4,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
        },

        # Encoding the inputs
        'main_encode_depth': 4,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
        },

        # Calculate moments
        'combine_moments': 'minus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_moments is 'concat'
        'combine_moments_encode_depth' : 1,
        'combine_moments_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },
        'moments_encode_depth' : 5,
        'moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    }
        },

        'moments_main_cond_encode_depth': 3,
        'moments_main_cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'pool': {'size' : 2, 'stride' : 2, 'type' : 'max'}
                    }
        },

        # Predict next moments
        'delta_moments_encode_depth' : 11,
        'delta_moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 0, 
                    'nonlinearity': nonlin},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 2, 
                    'nonlinearity': nonlin},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 4, 
                    'nonlinearity': nonlin},
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': {1: 14}, 
                    'nonlinearity': nonlin},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 0, 
                    'nonlinearity': nonlin},
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 2, 
                    'nonlinearity': nonlin},
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 4, 
                    'nonlinearity': nonlin},
                8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': {1: 18}, 
                    'nonlinearity': nonlin},
                9 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 0, 
                    'nonlinearity': nonlin},
                10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 2, 
                    'nonlinearity': nonlin},
                11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 4
                    },
        },
        'combine_delta': 'plus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_delta is 'concat'
        'combine_delta_encode_depth' : 1,
        'combine_delta_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },

        'next_main_encode_depth': 11,
        'next_main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 0, 
                    'nonlinearity': nonlin},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 2, 
                    'nonlinearity': nonlin},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 4, 
                    'nonlinearity': nonlin},
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': {1: 26}, 
                    'nonlinearity': nonlin},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 0, 
                    'nonlinearity': nonlin},
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 2, 
                    'nonlinearity': nonlin},
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 4, 
                    'nonlinearity': nonlin},
                8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': {1: 30}, 
                    'nonlinearity': nonlin},
                9 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 0, 
                    'nonlinearity': nonlin},
                10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 2, 
                    'nonlinearity': nonlin},
                11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'bypass': 4
                    },
        },

        'deconv_depth': 2,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                #'bypass': 4
                },
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                #'bypass': 2
                },
        }
}

def particle_cfg(n_classes, nonlin='relu'):
    return {
        # Encoding the inputs
        'main_encode_depth': 7,
        'main_encode' : {
            1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 150},
                'nonlinearity': nonlin
                },
            2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 150},
                'nonlinearity': nonlin
                },
            3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 150},
                'nonlinearity': nonlin
                },
            4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 150},
                'nonlinearity': nonlin
                },
            5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' :6*50},
                'nonlinearity': nonlin # up to 6 effects with each 50 dim
                },
            6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 100},
                'bypass': [0, 1], 'nonlinearity': nonlin
                # concat effects state and actions
                },
            7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : n_classes},
                'nonlinearity': nonlin
                }
        }
}

def cfg_mom_complete_bypass(n_classes, use_segmentation=True, 
        method='sign', nonlin='relu'):
    return {
        'use_segmentation': use_segmentation,
        # ONLY USED IF use_cond = True!!!
        'cond_scale_factor': 4,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
        },

        # Encoding the inputs
        'main_encode_depth': 4,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
        },

        # Calculate moments
        'combine_moments': 'minus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_moments is 'concat'
        'combine_moments_encode_depth' : 1,
        'combine_moments_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },
        'moments_encode_depth' : 5,
        'moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    }
        },

        'moments_main_cond_encode_depth': 3,
        'moments_main_cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'pool': {'size' : 2, 'stride' : 2, 'type' : 'max'}
                    }
        },

        # Predict next moments
        'delta_moments_encode_depth' : 11,
        'delta_moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 0, 'nonlinearity': nonlin},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 2, 'nonlinearity': nonlin},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4, 'nonlinearity': nonlin},
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': {1: 14}, 'nonlinearity': nonlin},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 0, 'nonlinearity': nonlin},
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 2, 'nonlinearity': nonlin},
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4, 'nonlinearity': nonlin},
                8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': {1: 18}, 'nonlinearity': nonlin},
                9 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 0, 'nonlinearity': nonlin},
                10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 2, 'nonlinearity': nonlin},
                11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4},
        },
        'combine_delta': 'plus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_delta is 'concat'
        'combine_delta_encode_depth' : 1,
        'combine_delta_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },

        'next_main_encode_depth': 11,
        'next_main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 0, 'nonlinearity': nonlin},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 2, 'nonlinearity': nonlin},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4, 'nonlinearity': nonlin},
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': {1: 26}, 'nonlinearity': nonlin},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 0, 'nonlinearity': nonlin},
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 2, 'nonlinearity': nonlin},
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4, 'nonlinearity': nonlin},
                8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': {1: 30}, 'nonlinearity': nonlin},
                9 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 0, 'nonlinearity': nonlin},
                10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 2, 'nonlinearity': nonlin},
                11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4},
        },

        'deconv_depth': 2,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                'bypass': 4
                },
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                'bypass': 2
                },
        }
}

def cfg_mom_flat_bypass(n_classes, use_cond=False, method='sign', nonlin='relu'):
    return {
        'use_cond': use_cond,
        # ONLY USED IF use_cond = True!!!
        'cond_scale_factor': 4,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
        },

        # Encoding the inputs
        'main_encode_depth': 4,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
        },

        # ONLY USED IF use_cond = True!!!
        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'pool': {'size' : 2, 'stride' : 2, 'type' : 'max'}
                    }
        },

        # Calculate moments
        'combine_moments': 'minus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_moments is 'concat'
        'combine_moments_encode_depth' : 1,
        'combine_moments_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },
        'moments_encode_depth' : 5,
        'moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    }
        },

        # Predict next moments
        'delta_moments_encode_depth' : 11,
        'delta_moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': [0,1], 'nonlinearity': nonlin},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4, 'nonlinearity': nonlin},
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 6, 'nonlinearity': nonlin},
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': [0,1], 'nonlinearity': nonlin},
                8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                9 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 4, 'nonlinearity': nonlin},
                10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'bypass': 6},

        },
        'combine_delta': 'plus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_delta is 'concat'
        'combine_delta_encode_depth' : 1,
        'combine_delta_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },

        'deconv_depth': 2,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                'bypass': 4},
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                'bypass': [0,1]},
        }
}

def cfg_mom_flat_concat(n_classes, use_cond=False, method='sign', nonlin='relu'):
    return {
        'use_cond': use_cond,
        # ONLY USED IF use_cond = True!!!
        'cond_scale_factor': 4,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
        },

        # Encoding the inputs
        'main_encode_depth': 4,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
        },

        # ONLY USED IF use_cond = True!!!
        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    #'pool': {'size' : 2, 'stride' : 2, 'type' : 'max'}
                    }
        },

        # Calculate moments
        'combine_moments': 'minus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_moments is 'concat'
        'combine_moments_encode_depth' : 1,
        'combine_moments_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },
        'moments_encode_depth' : 5,
        'moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    }
        },

        # Predict next moments
        'delta_moments_encode_depth' : 11,
        'delta_moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                9 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                10 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    'nonlinearity': nonlin},
                11 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },

        },
        'combine_delta': 'plus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_delta is 'concat'
        'combine_delta_encode_depth' : 1,
        'combine_delta_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 128},
                    }
        },

        'deconv_depth': 2,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                },
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                },
        }
}

def cfg_mom_concat(n_classes, use_cond=False, method='sign'):
    return {
        'use_cond': use_cond,
        # ONLY USED IF use_cond = True!!!
        'cond_scale_factor': 8,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
        },

        # Encoding the inputs
        'main_encode_depth': 8,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                    },
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                    },
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                8 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
        },

        # ONLY USED IF use_cond = True!!!
        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    #'pool': {'size' : 2, 'stride' : 2, 'type' : 'max'}
                    }
        },

        # Calculate moments
        'combine_moments': 'minus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_moments is 'concat'
        'combine_moments_encode_depth' : 1,
        'combine_moments_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 512},
                    }
        },
        'moments_encode_depth' : 5,
        'moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    }
        },

        # Predict next moments
        'delta_moments_encode_depth' : 5,
        'delta_moments_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    }
        },
        'combine_delta': 'plus' if method is 'sign' else 'concat',
        # ONLY USED IF combine_delta is 'concat'
        'combine_delta_encode_depth' : 1,
        'combine_delta_encode' : {
                1 : {'conv' : {'filter_size': 3, 'stride': 1, 'num_filters' : 512},
                    }
        },

        'deconv_depth': 3,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 512},
                },
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                },
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                },
            #4 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
            #    },
        }
}

def cfg_vgg_jerk_action(n_classes):
    return {
        'cond_scale_factor': 8,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
        },

        'main_encode_depth': 7,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 128},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                    },
                6 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                    },
                7 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                     'pool' : {'size' : 2, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_depth': 3,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    'pool': {'size' : 2, 'stride' : 2, 'type' : 'max'}}
        },

        'encode_together_depth' : 5,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    },
                5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 512},
                    }
        },
        'deconv_depth': 4,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 512},
                },
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                },
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 128},
                },
            4 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                },
        }
}

def cfg_wide_bypass_jerk_action(n_classes):
    return {
        'cond_scale_factor': 8,
        'cond_encode_depth': 1,
        'cond_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256}
                    },
        },

        'main_encode_depth': 3,
        'main_encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                    'bypass' : 0},
                3 : {'conv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                    'bypass' : 0},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_depth': 2,
        'encode' : {
                1 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256}
                    },
                2 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                    'bypass' : 0},
                #3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256},
                #    'bypass' : 0},
                    #'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
        },

        'encode_together_depth' : 2,
        'encode_together' : {
                1 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256},
                    'bypass' : 0},
                2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256},
                    'bypass' : 0},
                    #, 'bypass' : 0}
        },
        'hidden_depth': 0,

        'deconv_depth': 3,
        'deconv' : {
            1 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                'bypass' : 0},
            2 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : 256},
                'bypass' : 0},
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                'bypass' : 0},
        }
}

def cfg_bypass_jerk(n_classes):
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
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                'bypass' : 0},
        }
}

def cfg_sym_jerk(n_classes):
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
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes},
                'bypass' : 1},
        }
}

def cfg_map_jerk(n_classes):
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
            3 : {'deconv' : {'filter_size' : 3, 'stride' : 2, 'num_filters' : n_classes}},
        }
}

def cfg_res_jerk(n_classes):
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
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : n_classes}},
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






