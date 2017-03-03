"""
coupled symmetric model with from-below coupling
     --top-down is freely parameterized num-channels but from-below and top-down have same spatial extent 
     --top-down and bottom-up are combined via convolution to the correct num-channel shape:
        I = ReluConv(concat(top_down, bottom_up))
     --error is compuated as:
       (future_bottom_up - current_I)**2
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from curiosity.models.model_building_blocks import ConvNetwithBypasses
import curiosity.models.get_parameters as gp

DEBUG = True

def actionPredModel(inputs, min_time_difference, **kwargs):
    batch_size = inputs['images'].get_shape().as_list()[0]
    new_inputs = {'current' : inputs['images'], \
                  'poses' : inputs['poses'], \
                  'future_poses': inputs['future_poses'], \
                  'future' : inputs['future_images'], \
                  'times' : tf.ones([batch_size, min_time_difference])}
    return actionPredictionModelBase(new_inputs, **kwargs)

def actionPredictionModelBase(inputs, 
                        rng, 
                        cfg = {}, 
                        train = True, 
                        slippage = 0, 
                        minmax_end = True,
                        num_classes = None,
                        n_channels = 3,
                        **kwargs):
    '''
    Action Prediction Network Model Definition:

    current_images --> ENCODE -->  
                                CONCAT --> HIDDEN --> ACTIONS_PRED <-- ACTIONS_GT
    future_images  --> ENCODE --> 

    INPUT ARGUMENTS:
    inputs: {'current': images, 'future': images, 'actions': actions taken, 
             'times': time differences},
    rng: random number generator,
    cng: model definition,
    train: train model
    slippage: probability of random model changes
    min_max_end: use minmax clipping on last layer

    Outputs: dict with keys, pred and future, within those, dicts 
    with keys predi and futurei for i in 0:encode_depth, to be matched up in loss.
    '''


###### PREPROCESSING ######

    #set inputs
    current_node = inputs['current']
    future_node = inputs['future']
    actions_node = inputs['poses']
    time_node = inputs['times']

    #normalize and cast 
    current_node = tf.divide(tf.cast(current_node, tf.float32), 255)
    future_node = tf.divide(tf.cast(future_node, tf.float32), 255)
    actions_node = tf.cast(actions_node, tf.float32)
    original_actions = actions_node

    if(DEBUG):
        print('Actions shape')
        print(actions_node.get_shape().as_list())

    #init randon number generator
    if rng is None:
        rng = np.random.RandomState(seed=kwargs['seed'])

####### ENCODING #######

    #init from file
    init_file = '/media/data2/one_world_dataset/scripts/bvlc_alexnet_tfutils.npy'
    init_keys = [
     'conv1_b',
     'conv1_w',
     'conv2_b',
     'conv2_w',
     'conv3_b',
     'conv3_w',
     'conv4_b',
     'conv4_w',
     'conv5_b',
     'conv5_w',
     'fc6_b',
     'fc6_w',
     'fc7_b',
     'fc7_w',
     'fc8_b',
     'fc8_w']

    # necessary reshaping to use alexnet weights from caffe tensorflow conversion
    groups = [1, 2, 1, 2, 2]

    #init ConvNet
    net = ConvNetwithBypasses(**kwargs)

    encode_depth = gp.getEncodeDepth(rng, cfg, slippage=slippage)
  
    if(DEBUG):
        print('Encode depth: %d' % encode_depth)
  
    cfs0 = None

    # Split current nodes
    shape = current_node.get_shape().as_list()
    dim = int(shape[3] / n_channels)
    current_nodes = []
    for d in range(dim):
        current_nodes.append(tf.slice(current_node, [0,0,0,d*n_channels], [-1,-1,-1,n_channels]))

    #TODO Center crop current nodes to fit AlexNet
    for d in range(dim):
        current_nodes[d] = tf.image.resize_images(current_nodes[d], [192,192])
    future_node = tf.image.resize_images(future_node, [192,192])

    encode_nodes_current = [current_nodes]
    encode_nodes_future = [future_node]


    with tf.contrib.framework.arg_scope([net.conv, net.fc], \
                  init='xavier', stddev=.01, bias=0, activation='relu'): 
        for i in range(1, encode_depth + 1):
            with tf.variable_scope('encode' + str(i)) as encode_scope:
                #get encode parameters 
                #cfs: conv filter size, nf: number of filters, cs: conv stride
                #pfs: pool filter size, ps: pool stride, pool_type: pool type
                cfs, nf, cs, do_pool, pfs, ps, pool_type = gp.getEncodeParam(i, \
                        encode_depth, rng, cfg, prev=cfs0, slippage=slippage)
                cfs0 = cfs

                new_encode_nodes_current = []
                for encode_node_current in encode_nodes_current[i - 1]:
                    #encode current images (conv + pool)
                    init_layer_keys = {'bias': init_keys[2*(i-1)],
                                       'weight': init_keys[2*(i-1)+1]}
                    group = groups[(i-1)]
                    new_encode_node_current = net.conv(nf, cfs, cs, \
                            in_layer = encode_node_current, batch_normalize=False,
                            init='from_file', init_file=init_file, group=group, 
                            init_layer_keys=init_layer_keys, trainable=False)
                    if do_pool:
                        new_encode_node_current = net.pool(pfs, ps, \
                                in_layer = new_encode_node_current, pfunc = pool_type)
                    new_encode_nodes_current.append(new_encode_node_current)
                    #share the variables between current and future encoding
                    encode_scope.reuse_variables()
            
                #encode future images (conv + pool)
                new_encode_node_future = net.conv(nf, cfs, cs, \
                        in_layer = encode_nodes_future[i - 1], batch_normalize=False,
                        init='from_file', init_file=init_file, group=group,
                        init_layer_keys=init_layer_keys, trainable=False)
                if do_pool:
                    new_encode_node_future = net.pool(pfs, ps, \
                            in_layer = new_encode_node_future, pfunc = pool_type)
                    print('Pool size %d, stride %d' % (pfs, ps))
                    print('Type: ' + pool_type) 

                #store layers
                encode_nodes_current.append(new_encode_nodes_current)
                encode_nodes_future.append(new_encode_node_future)
 
                if(DEBUG):
                    print('Number of current node layers: ' + \
                            str(len(encode_nodes_current)))
                    print('Number of future node layers: ' + \
                            str(len(encode_nodes_future)))
                    print('Current encode node shape: ' + \
                            str(new_encode_node_current.get_shape().as_list()))
                    print('Future encode node shape: ' + \
                            str(new_encode_node_future.get_shape().as_list()))


###### HIDDEN ######

        encode_nodes_current = [encode_nodes_current[-1]]
        encode_nodes_future = [encode_nodes_future[-1]]

        '''
        #reshape to alexnet hidden dim
        encode_shape = encode_nodes_future[0].get_shape().as_list()
        fc6_alex_dim = 9216
        with tf.variable_scope('reshape' + str(i)) as reshape_scope:
            if(DEBUG):
                print('Linear from %d to %d' % (np.prod(encode_shape[1:]), fc6_alex_dim))
            
            new_encode_nodes_current = []
            for encode_node_current in encode_nodes_current[0]:
                new_encode_node_current = net.fc(fc6_alex_dim, bias = .01, \
                       in_layer = encode_node_current, \
                       activation = None, dropout = None)
                new_encode_nodes_current.append(new_encode_node_current)
                reshape_scope.reuse_variables()

            new_encode_node_future = net.fc(fc6_alex_dim, bias = .01, \
                       in_layer = encode_nodes_future[0], \
                       activation = None, dropout = None)

        encode_nodes_current = [new_encode_nodes_current]
        encode_nodes_future = [new_encode_node_future]
        '''

        #get hidden layer parameters
        nf0 = encode_nodes_future[-1].get_shape().as_list()[1]
        hidden_depth = gp.getHiddenDepth(rng, cfg, slippage=slippage)

        if(DEBUG):
            print('Hidden depth: %d' % hidden_depth)

        #fully connected hidden layers
        for i in range(1, hidden_depth + 1):
            with tf.variable_scope('hidden' + str(i)) as hidden_scope:

                nf = gp.getHiddenNumFeatures(i, hidden_depth, rng, \
                                             cfg, slippage=slippage)
    
                if(DEBUG):
                    print('Hidden shape %s' % nf)

                new_encode_nodes_current = []
                for encode_node_current in encode_nodes_current[i - 1]:
                    init_layer_keys = {'bias': init_keys[2*(i+4)],
                                       'weight': init_keys[2*(i+4)+1]}

                    new_encode_node_current = net.fc(nf, bias = 0.01, \
                            in_layer = encode_node_current, dropout = None, \
                            init='from_file', init_file=init_file, 
                            init_layer_keys=init_layer_keys, trainable=False)

                    new_encode_nodes_current.append(new_encode_node_current)
                    #share the variables between current and future encoding
                    hidden_scope.reuse_variables()

                nf0 = nf    

                #encode future images (conv + pool)
                new_encode_node_future = net.fc(nf, bias = 0.01, \
                        in_layer = encode_nodes_future[i - 1], dropout=None, \
                        init='from_file', init_file=init_file, 
                        init_layer_keys=init_layer_keys, trainable=False)

                #store layers
                encode_nodes_current.append(new_encode_nodes_current)
                encode_nodes_future.append(new_encode_node_future)

                if(DEBUG):
                    print('Number of current node layers: ' + \
                            str(len(encode_nodes_current)))
                    print('Number of future node layers: ' + \
                            str(len(encode_nodes_future)))
                    print('Current encode node shape: ' + \
                            str(new_encode_node_current.get_shape().as_list()))
                    print('Future encode node shape: ' + \
                            str(new_encode_node_future.get_shape().as_list()))


###### CONCAT ######

        with tf.variable_scope('concat'):
            #flatten
            flat_node_current = tf.concat(1, encode_nodes_current[-1])
            flat_node_current = flatten(net, flat_node_current)
            flat_node_future = flatten(net, encode_nodes_future[-1])

            #concat current and future
            encode_flat = tf.concat(1, [flat_node_current, flat_node_future])

        #match the shape of the action vector
        #by using another hidden layer if necessary
        ds = actions_node.get_shape().as_list()[1]

        if num_classes is not None:
            ds *= num_classes

        #TODO 
        if 'poses' in inputs:
            ds = num_classes

        if ds != nf0:
            with tf.variable_scope('extra_hidden'):
                pred = net.fc(ds, bias = .01, in_layer = encode_flat, \
                       activation = None, dropout = None)

            if(DEBUG):
                print("Linear from %d to %d" % (nf0, ds))
        else:
            pred = encode_flat

    if minmax_end:
        print("Min max clipping active")
        #pred = net.minmax(min_arg = 10, max_arg = -10, in_layer = pred)
        pred = tf.sigmoid(pred)
    #output batch normalized labels for storage and loss

    '''
    #first concatenate all actions into one batch batch to normalize across
    #the last two entries, i.e. the pixel values will always get normalized by 255
    norm_actions0 = []
    norm_actions1 = []
    for d in range(dim):
        norm_actions0.append(tf.slice(actions_node, [0, 8*d], [-1, 6]))
        norm_actions1.append(tf.slice(actions_node, [0, 6+8*d], [-1, 2]))
    norm_actions0 = tf.concat(0, norm_actions0)
    norm_actions1 = tf.concat(0, norm_actions1)

    #normalize
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(norm_actions0, [0])
    norm_actions0 = (norm_actions0 - batch_mean) / tf.sqrt(batch_var + epsilon)
    norm_actions1 = norm_actions1 / 255.0

    #reassemble actions vector
    batch_size = pred.get_shape().as_list()[0]
    norm_actions = []
    for d in range(dim):
        norm_actions.append(tf.slice(norm_actions0, [d*batch_size, 0], [batch_size, -1]))
        norm_actions.append(tf.slice(norm_actions1, [d*batch_size, 0], [batch_size, -1]))
    norm_actions = tf.concat(1, norm_actions)
    # normalize action vector
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(actions_node, [0])
    norm_actions = (actions_node - batch_mean) / tf.sqrt(batch_var + epsilon)
    ''' 

    theta, labels = get_quaternion_labels(inputs, dim, num_classes)

    outputs = {'pred': pred, 'theta_labels': labels, 'theta': theta}
    return outputs, net.params

def flatten(net, node):
    shape = node.get_shape().as_list()
    flat_node = net.reshape([np.prod(shape[1:])], in_layer = node)
    if(DEBUG):
        print('Flatten to shape %s' % flat_node.get_shape().as_list())
    return flat_node

def l2_action_loss(labels, logits, **kwargs):
    pred = logits['pred']
    shape = labels.get_shape().as_list()
    norm = shape[0] * shape[1]
    #batch normalized action features
    norm_labels = logits['norm_actions']
    #store normalized labels
    loss = tf.nn.l2_loss(pred - norm_labels) / norm    
    #loss = tf.minimum(loss, 0.1) #TODO remove and find reason!
    return loss

def binary_cross_entropy_action_loss(labels, logits, **kwargs):
    pred = logits['pred']
    dim = logits['dim']
    actions_node = logits['norm_actions']
    # Split action node and binary discretize
    action_shape = int(actions_node.get_shape().as_list()[1] / dim)
    actions_split = []
    for d in range(dim):
        act_sp = tf.slice(actions_node, [0, action_shape*d], [-1, action_shape])
#        act_sp = tf.reduce_sum(act_sp, 1)
        act_sp = tf.abs(act_sp)
        act_sp = tf.minimum(act_sp, 1)
        act_sp = tf.ceil(act_sp)        
#        act_sp = tf.reshape(act_sp, [-1, 1])
        actions_split.append(act_sp)
    norm_labels = tf.concat(1, actions_split)

    loss = -tf.reduce_sum(norm_labels * tf.log(pred) \
                          + (1 - norm_labels) * tf.log(1 - pred), 1)
    return loss


def get_quaternion_labels(inputs, seq_len, buckets_dim):
    shape = inputs['poses'].get_shape().as_list()
    dim = int(shape[1] / seq_len)

    assert dim == 4, 'dimension is not 4 but quaternions are used!'

    # upper bound for our 5 buckets, the last upper bound is inf and hence
    # not necessary
    assert buckets_dim == 3, 'Only num_classes=3 can be used with this loss.'

    # take the difference between the last and the first quaternion
    # quat_diff = quat_end * inv(quat_start)
    quat_start = tf.slice(inputs['poses'], [0, 0], [-1, dim])
    quat_end = inputs['future_poses']
    #inv(quat_start) = (q0 - iq1 - jq2 - kq3) / (q0^2 + q1^2 + q2^2 + q3^2)
    quats = []
    for i in range(dim):
        quats.append(tf.slice(quat_start, [0, i], [-1, 1]))

    quats[1] = tf.negative(quats[1])
    quats[2] = tf.negative(quats[2])
    quats[3] = tf.negative(quats[3]) 

    quat_norm = []
    for i in range(len(quats)):
        quat_norm.append(tf.multiply(quats[i], quats[i]))
    quat_norm = tf.concat(1, quat_norm)
    quat_norm = tf.reduce_sum(quat_norm, 1)
    quat_norm = tf.reshape(quat_norm, [-1, 1])

    quats = tf.concat(1, quats)

    inv_quat_start = tf.divide(quats, quat_norm)

    # quat multiplication: t = q * r
    # t0 = r0*q0 - r1*q1 - r2*q2 - r3*q3
    # t1 = r0*q1 + r1*q0 - r2*q3 + r3*q2
    # t2 = r0*q2 + r1*q3 + r2*q0 - r3*q1
    # t3 = r0*q3 - r1*q2 + r2*q1 + r3*q0
    r = []
    q = []
    t = []
    for i in range(dim):
        r.append(tf.slice(inv_quat_start, [0, i], [-1, 1]))
        q.append(tf.slice(quat_end, [0, i], [-1, 1])) 

    t.append(r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3])
    t.append(r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2])
    t.append(r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1])
    t.append(r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0])

    quat_diff = tf.concat(1, t) 

    # convert the quarternion to euler angle
    theta = tf.asin(2*(t[0]*t[2] - t[3]*t[1]))

    # bucket labels
    labels = []
    boundary = tf.constant(0.05, dtype=tf.float32)
    # 3 buckets: smaller zero, zero, greater zero
    labels.append(tf.less(theta, tf.negative(boundary)))
    labels.append(tf.logical_and(tf.greater(theta, tf.negative(boundary)),
                                 tf.less(theta, boundary)))
    labels.append(tf.greater(theta, boundary))

    # concatenate and cast to labels
    labels = tf.concat(1, labels)
    labels = tf.cast(labels, tf.int32)

    return [theta, labels]

def softmax_cross_entropy_pose_diff_loss(labels, logits, **kwargs):
    pred = logits['pred']
    labels = logits['theta_labels']
    # reshape predictions to match labels
    #pred = tf.reshape(pred, [-1,buckets_dim])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, labels))

    return loss
 
def softmax_cross_entropy_action_loss(labels, logits, **kwargs):
    pred = logits['pred']
    shape = labels.get_shape().as_list()
    norm_labels = logits['norm_actions']
    seq_len = logits['dim']
    dim = int(shape[1] / seq_len)

    '''
    #50:50 split
    pos_discount = np.array([10.13530233, 3.54033702, \
                             98.61325116, 2.72365054, \
                              1.92986523, 1.93214109])
    if dim == 4:
        pos_discount = pos_discount[0:4]

    rand = tf.random_uniform(shape)
    rands = []
    for i in range(seq_len):
        r = tf.slice(rand, [0, i*dim], [-1, dim])
        r = tf.multiply(r, pos_discount)
        r = tf.round(r)
        rands.append(r)
    rands = tf.concat(1, rands)
    rands = tf.cast(rands, tf.int32)
    '''

    #100:0 split, only consider pos examples
    rands = tf.ones(shape, dtype=tf.int32)

    # upper bound for our 5 buckets, the last upper bound is inf and hence
    # not necessary
    buckets_dim = logits['num_classes']
    if buckets_dim != 5:
        raise NotImplementedError('So far only num_classes=5 can be used \
              with this loss.')

    buckets = np.array(
              [[-5.0,  51.7, -156.1, -37.1, 141,  73], 
               [-4.9,  52.5,  -43.7,  -7.9, 178, 114], 
               [ 4.9,  52.6,   48.9,   7.5, 204, 140], 
               [ 5.1,  55.2,  166.7,  36.7, 230, 181]])
    if dim == 4:
        buckets = buckets[:,0:4]
    buckets = np.tile(buckets, (1,seq_len))

    # bucket labels
    labels = []
    new_shape = shape + [1]
    new_shape[0] = -1
    for i in range(buckets_dim):
        label = tf.zeros(shape, tf.bool)
        if i == 0:
            label = tf.less(norm_labels, buckets[i])
        elif i == buckets_dim - 1:
            label = tf.greater(norm_labels, buckets[i-1])
        else:
            label = tf.logical_and(tf.greater(norm_labels, buckets[i-1]), \
                                   tf.less(norm_labels, buckets[i]))
        label = tf.reshape(label, new_shape)
        labels.append(label)
    labels = tf.concat(2, labels)
    labels = tf.cast(labels, tf.int32)

    #find all positive indices
    zero = tf.constant(0, dtype=tf.float32)
    pos_idx = tf.not_equal(norm_labels, zero)
    neg_idx = tf.logical_not(pos_idx)        

    #mask out all negative entries
    rands = tf.reshape(rands, new_shape)
    pos_idx = tf.reshape(pos_idx, new_shape)
    neg_idx = tf.reshape(neg_idx, new_shape) 
    labels = labels * tf.cast(pos_idx, tf.int32) * rands \
           + labels * tf.cast(neg_idx, tf.int32) * (1-rands)

    pred = tf.reshape(pred, [-1,buckets_dim])
    #labels = tf.reshape(labels, [-1,5])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, labels))
    return loss
