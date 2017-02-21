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
                  'actions' : inputs['parsed_actions'], \
                  'future' : inputs['future_images'], \
                  'times' : tf.ones([batch_size, min_time_difference])}
    return actionPredictionModelBase(new_inputs, **kwargs)

def preprocessing(inputs,
                  n_channels = 3):
    raise NotImplementedError

def nNet(inputs,
            rng,
            cfg = {},
            train = True,
            slippage = 0,
            n_channels = 3,
            **kwargs):
    raise NotImplementedError

def actionPredictionModelBase(inputs, 
                        rng, 
                        cfg = {}, 
                        train = True, 
                        slippage = 0, 
                        minmax_end = True,
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
    actions_node = inputs['actions']
    time_node = inputs['times']

    #normalize and cast 
    current_node = tf.divide(tf.cast(current_node, tf.float32), 255)
    future_node = tf.divide(tf.cast(future_node, tf.float32), 255)
    actions_node = tf.cast(actions_node, tf.float32)

    if(DEBUG):
        print('Actions shape')
        print(actions_node.get_shape().as_list())

    #init randon number generator
    if rng is None:
        rng = np.random.RandomState(seed=kwargs['seed'])

    #init ConvNet
    net = ConvNetwithBypasses(**kwargs)


####### ENCODING #######

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

    encode_nodes_current = [current_nodes]
    encode_nodes_future = [future_node]

    with tf.contrib.framework.arg_scope([net.conv, net.fc], \
                  init='trunc_norm', stddev=.01, bias=0, activation='relu', \
                  ): 
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
                    new_encode_node_current = net.conv(nf, cfs, cs, \
                            in_layer = encode_node_current, batch_normalize=True)
                    if do_pool:
                        new_encode_node_current = net.pool(pfs, ps, \
                                in_layer = new_encode_node_current, pfunc = pool_type)
                    new_encode_nodes_current.append(new_encode_node_current)
                    #share the variables between current and future encoding
                    encode_scope.reuse_variables()
            
                #encode future images (conv + pool)
                new_encode_node_future = net.conv(nf, cfs, cs, \
                        in_layer = encode_nodes_future[i - 1], batch_normalize=True)
                if do_pool:
                    new_encode_node_future = net.pool(pfs, ps, \
                            in_layer = new_encode_node_future, pfunc = pool_type)
 
                if(DEBUG):
                    print('Current encode node shape: ' + \
                            str(new_encode_node_current.get_shape().as_list()))
                    print('Future encode node shape: ' + \
                            str(new_encode_node_current.get_shape().as_list()))
                    print('Pool size %d, stride %d' % (pfs, ps))
                    print('Type: ' + pool_type) 

                #store layers
                encode_nodes_current.append(new_encode_nodes_current)
                encode_nodes_future.append(new_encode_node_future)


###### HIDDEN ######

        with tf.variable_scope('concat'):
            #flatten
            flat_node_current = tf.concat(3, encode_nodes_current[-1])
            flat_node_current = flatten(net, flat_node_current)
            flat_node_future = flatten(net, encode_nodes_future[-1])

            #concat current and future
            encode_flat = tf.concat(1, [flat_node_current, flat_node_future])

            #get hidden layer parameters
            nf0 = encode_flat.get_shape().as_list()[1]
            hidden_depth = gp.getHiddenDepth(rng, cfg, slippage=slippage)
  
            if(DEBUG):
                print('Hidden depth: %d' % hidden_depth)

        #fully connected hidden layers
        hidden = encode_flat
        for i in range(1, hidden_depth + 1):
            with tf.variable_scope('hidden' + str(i)):
                nf = gp.getHiddenNumFeatures(i, hidden_depth, rng, cfg, slippage=slippage)
                hidden = net.fc(nf, bias = 0.01, in_layer = hidden, dropout = None)
                nf0 = nf 
    
                if(DEBUG):
                    print('Hidden shape %s' % hidden.get_shape().as_list())

#        #TODO Test and remove summing actions
#        shape = actions_node.get_shape().as_list()
#        actions_node = net.reshape([6,int(shape[1]/6)], in_layer = actions_node)
#        actions_node = tf.reduce_sum(actions_node, 2)

        #match the shape of the action vector
        #by using another hidden layer if necessary
        ds = actions_node.get_shape().as_list()[1]
        if ds != nf0:
            with tf.variable_scope('extra_hidden'):
                pred = net.fc(ds, bias = .01, activation = None, dropout = None)

            if(DEBUG):
                print("Linear from %d to %d" % (nf0, ds))
        else:
            pred = hidden

    if minmax_end:
        print("Min max clipping active")
        #pred = net.minmax(min_arg = 10, max_arg = -10, in_layer = pred)
        pred = tf.tanh(pred) * 10

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
    actions_mean = np.array(
      [  3.90625000e-03,   0.00000000e+00,  -2.44093345e-04, #teleport, vel_x, vel_y
         5.20533161e-02,   1.22121812e+02,  -3.34609385e-05, #vel_z, ang_x, ang_y
        -7.55522251e+02,  -1.47485679e-01,   1.06664636e+02, #ang_z, a1_fx, a1_fy
         1.28125378e-02,   0.00000000e+00,  -2.27804319e-02, #a1_fz, a1_tx, a1_ty,
         0.00000000e+00,   3.44634385e+01,   7.06908594e+01, #a1_tz, a1_id, a1_pos_x,
         6.58260645e+01,   3.78274760e-02,   0.00000000e+00, #a1_pos_y, a2_fx, a2_fy,
        -1.28125378e-02,   0.00000000e+00,  -1.32991500e-03, #a2_fz, a2_tx, a2_ty,
         0.00000000e+00,   6.64200195e-01,   1.27456543e+00, #a2_tz, a2_id, a2_pos_x,
         1.30035547e+00]).astype(np.float32)                 #a2_pos_y

    actions_std = np.array(
      [  6.23778102e-02,   1+0.00000000e+00,   4.53425576e-03,
         1.01547240e-01,   2.22054444e+06,   6.04687621e-02,
         1.43378085e+06,   1.27678463e+02,   3.23207041e+03,
         1.95972036e+01,   1+0.00000000e+00,   1.37277482e+01,
         1+0.00000000e+00,   6.96205264e+01,   1.26656184e+02,
         1.27864069e+02,   2.00925928e+01,   1+0.00000000e+00,
         1.95972036e+01,   1+0.00000000e+00,   6.21731960e-01,
         1+0.00000000e+00,   1.07432982e+01,   1.84946833e+01,
         2.16857321e+01]).astype(np.float32)

    actions_min = np.array(
       [  0.00000000e+00,   0.00000000e+00,  -8.44568036e-02,
         -1.96909573e-01,  -2.37563629e+09,  -9.25147434e+00,
         -2.00465812e+09,  -1.02400000e+04,   0.00000000e+00,
         -1.53459117e+03,   0.00000000e+00,  -4.99995988e+01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,  -1.90427973e+03,   0.00000000e+00,
         -1.64339652e+03,   0.00000000e+00,  -1.49970194e+01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00]
    ).astype(np.float32)

    actions_max = np.array(
       [  1.00000000e+00,   1+0.00000000e+00,   1.05496711e-01,
          2.50000000e-01,   2.04533410e+09,   1.82908107e+00,
          4.27925173e+08,   1.02400000e+04,   2.15040000e+05,
          1.64339652e+03,   1+0.00000000e+00,   4.99999369e+01,
          1+0.00000000e+00,   4.40000000e+02,   3.83000000e+02,
          5.11000000e+02,   1.61245106e+03,   1+0.00000000e+00,
          1.53459117e+03,   1+0.00000000e+00,   1.49993386e+01,
          1+0.00000000e+00,   3.85000000e+02,   3.83000000e+02,
          5.11000000e+02]
    ).astype(np.float32)

    actions_std = np.concatenate([actions_std[7:13], actions_std[14:16]], axis=0)
    actions_mean = np.concatenate([actions_mean[7:13], actions_mean[14:16]], axis=0)
    actions_min = np.concatenate([actions_min[7:13], actions_min[14:16]], axis=0)
    actions_max = np.concatenate([actions_max[7:13], actions_max[14:16]], axis=0)

    actions_std = np.tile(actions_std, dim)
    actions_mean = np.tile(actions_mean, dim)
    actions_min = np.tile(actions_min, dim)
    actions_max = np.tile(actions_max, dim)

#    norm_actions = (actions_node - actions_mean) / actions_std
    norm_actions = (actions_node - actions_min) / (actions_max - actions_min) 

    outputs = {'pred': pred, 'norm_actions': norm_actions}
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
