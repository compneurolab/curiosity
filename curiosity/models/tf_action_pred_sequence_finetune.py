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
import cPickle
from scipy.misc import imresize

DEBUG = True

global dataset_stats 
dataset_stats = None 

def load_stats(n_channels):
    global dataset_stats

    print('Loading data statistics...')

    stats_file = '/media/data/one_world_dataset/dataset_stats.pkl' 
    image_size = [256, 256, 3]
    with open(stats_file) as f:
        stats = cPickle.load(f)

    for k in ['mean', 'std']:
        stats[k]['images'] = \
            stats[k]['images'].astype(np.float32)
        stats[k]['images'] = \
            imresize(stats[k]['images'], \
            image_size).astype(np.float32)

    if n_channels != 3:
        raise NotImplementedError

    dataset_stats = stats

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
                        normalize_images_after_dequeue = False,
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
    current_node = tf.cast(current_node, tf.float32)
    future_node = tf.cast(future_node, tf.float32)
    #current_node = tf.divide(tf.cast(current_node, tf.float32), 255)
    #future_node = tf.divide(tf.cast(future_node, tf.float32), 255)
    actions_node = tf.cast(actions_node, tf.float32)

    if(DEBUG):
        print('Actions shape')
        print(actions_node.get_shape().as_list())

    #init randon number generator
    if rng is None:
        rng = np.random.RandomState(seed=kwargs['seed'])

    # Split current nodes
    shape = current_node.get_shape().as_list()
    dim = int(shape[3] / n_channels)
    current_nodes = []
    for d in range(dim):
        current_nodes.append(tf.slice(current_node, \
            [0,0,0,d*n_channels], [-1,-1,-1,n_channels]))

    if normalize_images_after_dequeue:
        if(DEBUG):
            print('Normalizing images')
        # load dataset_stats
        if dataset_stats is None:
            load_stats(n_channels)

        #normalize images
        future_node = (future_node - dataset_stats['mean']['images']) / \
            (dataset_stats['std']['images'] + 10e-4)

        for d in range(dim):
            current_nodes[d] = (current_nodes[d] - dataset_stats['mean']['images']) / \
                (dataset_stats['std']['images'] + 10e-4)

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

    encode_nodes_current = [current_nodes]
    encode_nodes_future = [future_node]

    with tf.contrib.framework.arg_scope([net.conv, net.fc], \
                  init='xavier', stddev=.01, bias=0, activation='relu', \
                  ): 
        for i in range(1, encode_depth + 1):
            with tf.variable_scope('encode' + str(i)) as encode_scope:
                #get encode parameters 
                #cfs: conv filter size, nf: number of filters, cs: conv stride
                #pfs: pool filter size, ps: pool stride, pool_type: pool type
                cfs, nf, cs, do_pool, pfs, ps, pool_type = gp.getEncodeParam(i, \
                        encode_depth, rng, cfg, prev=cfs0, slippage=slippage)
                cfs0 = cfs

                #print( cfs, nf, cs, do_pool, pfs, ps, pool_type)
                #print(init_keys[2*(i-1)], init_keys[2*(i-1)+1])

                new_encode_nodes_current = []
                for encode_node_current in encode_nodes_current[i - 1]:
                    #print(encode_node_current)
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
        pred = tf.tanh(pred) #*10

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
    norm_actions = actions_node
    outputs = {'pred': pred, 'norm_actions': norm_actions, 'dim': dim}
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

def weighted_l2_action_loss(labels, logits, **kwargs):
    pred = logits['pred']
    shape = labels.get_shape().as_list()
    norm = shape[0] * shape[1]
    norm_labels = logits['norm_actions']
    seq_len = logits['dim']    
    dim = int(shape[1] / seq_len) 

    #50:50 split
    pos_discount = np.array([10.13530233, 3.54033702, \
                             98.61325116, 2.72365054, \
                              1.92986523, 1.93214109])
#    neg_discount = np.array([0.52594625, 0.58222773, \
#                             0.50254808, 0.61242774, \
#                             0.67484165, 0.67456381])
    if dim == 4:
        pos_discount = pos_discount[0:4]
#        neg_discount = neg_discount[0:4]

    rand = tf.random_uniform(shape)
    rands = []
    for i in range(seq_len):
        r = tf.slice(rand, [0, i*dim], [-1, dim])
        r = tf.multiply(r, pos_discount)
        r = tf.round(r)
        rands.append(r)
    rands = tf.concat(1, rands)

    #TODO REMOVE: only consider pos examples
    rands = tf.ones(shape)

    zero = tf.constant(0, dtype=tf.float32)
    pos_idx = tf.not_equal(norm_labels, zero)
    neg_idx = tf.logical_not(pos_idx)

    diff = pred - norm_labels
    diff = diff * tf.cast(pos_idx, tf.float32) * rands \
         + diff * tf.cast(neg_idx, tf.float32) * (1-rands)
    loss = tf.nn.l2_loss(diff) / norm
    return loss
