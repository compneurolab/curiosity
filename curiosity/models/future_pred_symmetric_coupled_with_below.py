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
import zmq
#hrm, check out paths
from curiosity.models.model_building_blocks import ConvNetwithBypasses

from curiosity.utils.io import recv_array

ctx = zmq.Context()
sock = None
IMAGE_SIZE = None
NUM_CHANNELS = None
ACTION_LENGTH = None

def initialize(host, port, datapath, keyname):
  global ctx, sock, IMAGE_SIZE, NUM_CHANNELS, ACTION_LENGTH
  sock = ctx.socket(zmq.REQ)
  print("connecting...")
  sock.connect("tcp://%s:%d" % (host, port))
  print("...connected")
  sock.send_json({'batch_size': 1,
                  'batch_num': 0,
                  'path': datapath,
                  'keys': [(keyname, 'images0'), 
                           (keyname, 'actions')]
                 })
  images = recv_array(sock)
  actions = recv_array(sock)
  IMAGE_SIZE = images.shape[1]
  NUM_CHANNELS = images.shape[-1]
  ACTION_LENGTH = actions.shape[1]


def getEncodeDepth(rng, cfg, slippage=0):
  val = None
  if 'encode_depth' in cfg:
    val = cfg['encode_depth']
  elif 'encode' in cfg:
    val = max(cfg['encode'].keys())
  if val is not None and rng.uniform() > slippage:
    return val
  d = rng.choice([1, 2, 3, 4, 5])
  return d

def getEncodeConvFilterSize(i, encode_depth, rng, cfg, prev=None, slippage=0):
  val = None
  if 'encode' in cfg and (i in cfg['encode']):
    if 'conv' in cfg['encode'][i]:
      if 'filter_size' in cfg['encode'][i]['conv']:
        val = cfg['encode'][i]['conv']['filter_size']  
  if val is not None and rng.uniform() > slippage:
    return val
  L = [1, 3, 5, 7, 9, 11, 13, 15, 23]
  if prev is not None:
    L = [_l for _l in L if _l <= prev]
  return rng.choice(L)

def getEncodeConvNumFilters(i, encode_depth, rng, cfg, slippage=0):
  val = None
  if 'encode' in cfg and (i in cfg['encode']):
    if 'conv' in cfg['encode'][i]:
      if 'num_filters' in cfg['encode'][i]['conv']:
        val = cfg['encode'][i]['conv']['num_filters']
  if val is not None and rng.uniform() > slippage:
    return val
  L = [3, 48, 96, 128, 256, 128]
  return L[i]
  
def getEncodeConvStride(i, encode_depth, rng, cfg, slippage=0):
  val = None
  if 'encode' in cfg and (i in cfg['encode']):
    if 'conv' in cfg['encode'][i]:
      if 'stride' in cfg['encode'][i]['conv']:
        val = cfg['encode'][i]['conv']['stride']
  if val is not None and rng.uniform() > slippage:
    return val
  if encode_depth > 1:
    return 2 if i == 1 else 1
  else:
    return 3 if i == 1 else 1

def getEncodeDoPool(i, encode_depth, rng, cfg, slippage=0):
  val = None
  if 'encode' in cfg and (i in cfg['encode']):
    if 'do_pool' in cfg['encode'][i]:
      val = cfg['encode'][i]['do_pool']
    elif 'pool' in cfg['encode'][i]:
      val = True
  if val is not None and rng.uniform() > slippage:
    return val
  if i < 3 or i == encode_depth:
    return rng.uniform() < .75
  else:
    return rng.uniform() < .25
    
def getEncodePoolFilterSize(i, encode_depth, rng, cfg, slippage=0):
  val = None
  if 'encode' in cfg and (i in cfg['encode']):
    if 'pool' in cfg['encode'][i]:
      if 'filter_size' in cfg['encode'][i]['pool']:
        val = cfg['encode'][i]['pool']['filter_size']
  if val is not None and rng.uniform() > slippage:
    return val
  return rng.choice([2, 3, 4, 5])

def getEncodePoolStride(i, encode_depth, rng, cfg, slippage=0):  
  val = None
  if 'encode' in cfg and (i in cfg['encode']):
    if 'pool' in cfg['encode'][i]:
      if 'stride' in cfg['encode'][i]['pool']:
        val = cfg['encode'][i]['pool']['stride']
  if val is not None and rng.uniform() > slippage:
    return val
  return 2

def getEncodePoolType(i, encode_depth, rng, cfg, slippage=0):
  val = None
  if 'encode' in cfg and (i in cfg['encode']):
    if 'pool' in cfg['encode'][i]:
      if 'type' in cfg['encode'][i]['pool']:
        val = cfg['encode'][i]['pool']['type']
  if val is not None and rng.uniform() > slippage:
    return val
  return rng.choice(['max', 'avg'])

def getHiddenDepth(rng, cfg, slippage=0):
  val = None
  if (not rng.uniform() < slippage) and 'hidden_depth' in cfg:
    val = cfg['hidden_depth']
  elif 'hidden' in cfg:
    val = max(cfg['hidden'].keys())
  if val is not None and rng.uniform() > slippage:
    return val
  d = rng.choice([1, 2, 3])
  return d

def getHiddenNumFeatures(i, hidden_depth, rng, cfg, slippage=0):
  val = None
  if 'hidden' in cfg and (i in cfg['hidden']):
    if 'num_features' in cfg['hidden'][i]:
      val = cfg['hidden'][i]['num_features']
  if val is not None and rng.uniform() > slippage:
    return val
  return 1024

def getDecodeDepth(rng, cfg, slippage=0):
  val = None
  if 'decode_depth' in cfg:
    val = cfg['decode_depth']
  elif 'decode' in cfg:
    val = max(cfg['decode'].keys())
  if val is not None and rng.uniform() > slippage:
    return val
  d = rng.choice([1, 2, 3])
  return d

def getDecodeNumFilters(i, decode_depth, rng, cfg, slippage=0):
  val = None
  if 'decode' in cfg and (i in cfg['decode']):
    if 'num_filters' in cfg['decode'][i]:
      val = cfg['decode'][i]['num_filters']
  if val is not None and rng.uniform() > slippage:
    return val
  return 32

def getDecodeFilterSize(i, decode_depth, rng, cfg, slippage=0):
  val = None
  if 'decode' in cfg and (i in cfg['decode']):
     if 'filter_size' in cfg['decode'][i]:
       val = cfg['decode'][i]['filter_size']
  if val is not None and rng.uniform() > slippage:
    return val
  return rng.choice([1, 3, 5, 7, 9, 11])
  
def getDecodeFilterSize2(i, decode_depth, rng, cfg, slippage=0):
  val = None
  if 'decode' in cfg and (i in cfg['decode']):
     if 'filter_size2' in cfg['decode'][i]:
       val = cfg['decode'][i]['filter_size2']
  if val is not None and rng.uniform() > slippage:
    return val
  return rng.choice([1, 3, 5, 7, 9, 11])

def getDecodeSize(i, decode_depth, init, final, rng, cfg, slippage=0):
  val = None
  if 'decode' in cfg and (i in cfg['decode']):
    if 'size' in cfg['decode'][i]:
      val = cfg['decode'][i]['size']
  if val is not None and rng.uniform() > slippage:
    return val
  s = np.log2(init)
  e = np.log2(final)
  increment = (e - s) / decode_depth
  l = np.around(np.power(2, np.arange(s, e, increment)))
  if len(l) < decode_depth + 1:
    l = np.concatenate([l, [final]])
  l = l.astype(np.int)
  return l[i]

def getDecodeBypass(i, encode_nodes, decode_size, decode_depth, rng, cfg, slippage=0):
  val = None
  if 'decode' in cfg and (i in cfg['decode']):
    if 'bypass' in cfg['decode'][i]:
      val = cfg['decode'][i]['bypass']
  #prevent error that can occur here if encode is not large enough due to slippage modification?
  if val is not None and rng.uniform() > slippage:
    return val 
  switch = rng.uniform() 
  print('sw', switch)
  if switch < 0.5:
    sdiffs = [e.get_shape().as_list()[1] - decode_size for e in encode_nodes]
    return np.abs(sdiffs).argmin()
    
def getFilterSeed(rng, cfg):
  if 'filter_seed' in cfg:
    return cfg['filter_seed']
  else:  
    return rng.randint(10000)


def model_tfutils_fpd_compatible(inputs, **kwargs):
  batch_size = inputs['images'].get_shape().as_list()[0]
  new_inputs = {'current' : inputs['images'], 'actions' : inputs['actions'], 'future' : inputs['future_images'], 'time' : tf.ones([batch_size, 1])}
  return model_tfutils(new_inputs, **kwargs)


def model_tfutils(inputs, rng, cfg = {}, train = True, slippage = 0, **kwargs):
  '''Model definition, compatible with tfutils.

  inputs should have 'current', 'future', 'action', 'time' keys. Outputs is a dict with keys, pred and future, within those, dicts with keys predi and futurei for i in 0:encode_depth, to be matched up in loss.'''
  current_node = inputs['current']
  future_node = inputs['future']
  actions_node = inputs['actions']
  time_node = inputs['time']
  print('data types')
  for k, node in inputs.iteritems():
    print(k)
    print(node.dtype)
    print(node.get_shape().as_list())

#I think this should be taken away from cfg
  # fseed = getFilterSeed(rng, cfg)

  if rng is None:
    rng = np.random.RandomState(seed=kwargs['seed'])

  m = ConvNetwithBypasses(**kwargs)

  #encoding
  encode_depth = getEncodeDepth(rng, cfg, slippage=slippage)
  print('Encode depth: %d' % encode_depth)
  cfs0 = None

  encode_nodes_current = [current_node]
  encode_nodes_future = [future_node]
  for i in range(1, encode_depth + 1):
    #not sure this usage ConvNet class creates exactly the params that we want to have, specifically in the 'input' field, but should give us an accurate record of this network's configuration
    with tf.variable_scope('encode' + str(i)):

      with tf.contrib.framework.arg_scope([m.conv], init='trunc_norm', stddev=.01, bias=0, activation='relu'):

        cfs = getEncodeConvFilterSize(i, encode_depth, rng, cfg, prev=cfs0, slippage=slippage)
        cfs0 = cfs
        nf = getEncodeConvNumFilters(i, encode_depth, rng, cfg, slippage=slippage)
        cs = getEncodeConvStride(i, encode_depth, rng, cfg, slippage=slippage)

        new_encode_node_current = m.conv(nf, cfs, cs, in_layer = encode_nodes_current[i - 1])
    with tf.variable_scope('encode' + str(i), reuse = True):
      new_encode_node_future = m.conv(nf, cfs, cs, in_layer = encode_nodes_future[i - 1], init='trunc_norm', stddev=.01, bias=0, activation='relu')
  #TODO add print function
      do_pool = getEncodeDoPool(i, encode_depth, rng, cfg, slippage=slippage)
      if do_pool:
        pfs = getEncodePoolFilterSize(i, encode_depth, rng, cfg, slippage=slippage)
        ps = getEncodePoolStride(i, encode_depth, rng, cfg, slippage=slippage)
        pool_type = getEncodePoolType(i, encode_depth, rng, cfg, slippage=slippage)
        new_encode_node_current = m.pool(pfs, ps, in_layer = new_encode_node_current)
        new_encode_node_future = m.pool(pfs, ps, in_layer = new_encode_node_future)

      encode_nodes_current.append(new_encode_node_current)
      encode_nodes_future.append(new_encode_node_future)

#TODO: change variable scope when param update is introduced
  encode_node = encode_nodes_current[-1]
  enc_shape = encode_node.get_shape().as_list()
  encode_flat = m.reshape([np.prod(enc_shape[1:])], in_layer = encode_node)
  print('Flatten to shape %s' % encode_flat.get_shape().as_list())
#TODO: add functionality to extension to deal with this
  if time_node is not None:
    encode_flat = tf.concat(1, [encode_flat, actions_node, time_node])
  else:
    encode_flat = tf.concat(1, [encode_flat, actions_node])

  nf0 = encode_flat.get_shape().as_list()[1]
  hidden_depth = getHiddenDepth(rng, cfg, slippage=slippage)
  print('Hidden depth: %d' % hidden_depth)
  hidden = encode_flat
  for i in range(1, hidden_depth + 1):
    with tf.variable_scope('hidden' + str(i)):
      nf = getHiddenNumFeatures(i, hidden_depth, rng, cfg, slippage=slippage)
      #TODO: this can be made nicer once we add more general concat
      hidden = m.fc(nf, init = 'trunc_norm', activation = 'relu', bias = .01, in_layer = hidden, dropout = None)



  #decode
  ds = encode_nodes_future[encode_depth].get_shape().as_list()[1]
  nf1 = getDecodeNumFilters(0, encode_depth, rng, cfg, slippage=slippage)
  if ds * ds * nf1 != nf0:
    with tf.variable_scope('extra_hidden'):
      hidden = m.fc(ds * ds * nf1, init = 'trunc_norm', activation  = None, bias = .01, dropout = None)
    print("Linear from %d to %d for input size %d" % (nf0, ds * ds * nf1, ds))
  decode = m.reshape([ds, ds, nf1])
  print("Unflattening to", decode.get_shape().as_list())



  preds = {}
  for i in range(0, encode_depth + 1):
    with tf.variable_scope('pred' + str(encode_depth - i)):
      pred = m.add_bypass(encode_nodes_current[encode_depth - i])
      nf = encode_nodes_future[encode_depth - i].get_shape().as_list()[-1]
      cfs = getDecodeFilterSize2(i, encode_depth, rng, cfg, slippage = slippage)
      if i == encode_depth:
        #TODO: add functionality so that this architecture is reflected in params        
        pred = m.conv(nf, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation=None)
        pred = tf.minimum(tf.maximum(pred, -1), 1)
      else:
        pred = m.conv(nf, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation='relu')
      preds['pred' + str(encode_depth - i)] = pred
    if i != encode_depth:
      with tf.variable_scope('decode' + str(i+1)):
        ds = encode_nodes_future[encode_depth - i - 1].get_shape().as_list()[1]
        decode = m.resize_images(ds, in_layer = decode)
        print('Decode resize %d to shape' % (i + 1), decode.get_shape().as_list())
        cfs = getDecodeFilterSize(i + 1, encode_depth, rng, cfg, slippage=slippage)
        nf1 = getDecodeNumFilters(i + 1, encode_depth, rng, cfg, slippage=slippage)
        decode = m.conv(nf1, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation='relu')

  encode_nodes_future_dict = dict(('future' + str(i), encoded_future) for (i, encoded_future) in enumerate(encode_nodes_future))
  outputs = {'pred' : preds, 'future' : encode_nodes_future_dict}

  return outputs, m.params







def model(current_node, future_node, actions_node, time_node, rng, cfg, slippage=0, slippage_error=False):
  """The Model definition."""
  cfg0 = {} 

  fseed = getFilterSeed(rng, cfg)
  
  #encoding
  nf0 = NUM_CHANNELS
  imsize = IMAGE_SIZE
  encode_depth = getEncodeDepth(rng, cfg, slippage=slippage)
  cfg0['encode_depth'] = encode_depth
  print('Encode depth: %d' % encode_depth)
  encode_nodes_current = []
  encode_nodes_current.append(current_node)
  encode_nodes_future = []
  encode_nodes_future.append(future_node)
  cfs0 = None
  cfg0['encode'] = {}
  for i in range(1, encode_depth + 1):
    cfg0['encode'][i] = {}
    cfs = getEncodeConvFilterSize(i, encode_depth, rng, cfg, prev=cfs0, slippage=slippage)
    cfg0['encode'][i]['conv'] = {'filter_size': cfs}
    cfs0 = cfs
    nf = getEncodeConvNumFilters(i, encode_depth, rng, cfg, slippage=slippage)
    cfg0['encode'][i]['conv']['num_filters'] = nf
    cs = getEncodeConvStride(i, encode_depth, rng, cfg, slippage=slippage)
    cfg0['encode'][i]['conv']['stride'] = cs
    W = tf.Variable(tf.truncated_normal([cfs, cfs, nf0, nf],
                                        stddev=0.01,
                                        seed=fseed))
    new_encode_node_current = tf.nn.conv2d(encode_nodes_current[i-1], W,
                               strides = [1, cs, cs, 1],
                               padding='SAME')
    new_encode_node_current = tf.nn.relu(new_encode_node_current)
    new_encode_node_future = tf.nn.conv2d(encode_nodes_future[i-1], W,
                               strides = [1, cs, cs, 1],
                               padding='SAME')
    new_encode_node_future = tf.nn.relu(new_encode_node_future)
    b = tf.Variable(tf.zeros([nf]))
    new_encode_node_current = tf.nn.bias_add(new_encode_node_current, b)
    new_encode_node_future = tf.nn.bias_add(new_encode_node_future, b)
    imsize = imsize // cs
    print('Encode conv %d with size %d stride %d num channels %d numfilters %d for shape' % (i, cfs, cs, nf0, nf), new_encode_node_current.get_shape().as_list())    
    do_pool = getEncodeDoPool(i, encode_depth, rng, cfg, slippage=slippage)
    if do_pool:
      pfs = getEncodePoolFilterSize(i, encode_depth, rng, cfg, slippage=slippage)
      cfg0['encode'][i]['pool'] = {'filter_size': pfs}
      ps = getEncodePoolStride(i, encode_depth, rng, cfg, slippage=slippage)
      cfg0['encode'][i]['pool']['stride'] = ps
      pool_type = getEncodePoolType(i, encode_depth, rng, cfg, slippage=slippage)
      cfg0['encode'][i]['pool']['type'] = pool_type
      if pool_type == 'max':
        pfunc = tf.nn.max_pool
      elif pool_type == 'avg':
        pfunc = tf.nn.avg_pool
      new_encode_node_current = pfunc(new_encode_node_current,
                          ksize = [1, pfs, pfs, 1],
                          strides = [1, ps, ps, 1],
                          padding='SAME')
      new_encode_node_future = pfunc(new_encode_node_future,
                          ksize = [1, pfs, pfs, 1],
                          strides = [1, ps, ps, 1],
                          padding='SAME')                        
      print('Encode %s pool %d with size %d stride %d for shape' % (pool_type, i, pfs, ps),
                    new_encode_node_current.get_shape().as_list())
      imsize = imsize // ps
    nf0 = nf

    encode_nodes_current.append(new_encode_node_current)   
    encode_nodes_future.append(new_encode_node_future)

  encode_node = encode_nodes_current[-1]
  enc_shape = encode_node.get_shape().as_list()
  encode_flat = tf.reshape(encode_node, [enc_shape[0], np.prod(enc_shape[1:])])
  print('Flatten to shape %s' % encode_flat.get_shape().as_list())

  encode_flat = tf.concat(1, [encode_flat, actions_node, time_node]) 
  #hidden
  nf0 = encode_flat.get_shape().as_list()[1]
  hidden_depth = getHiddenDepth(rng, cfg, slippage=slippage)
  cfg0['hidden_depth'] = hidden_depth
  hidden = encode_flat
  cfg0['hidden'] = {}
  for i in range(1, hidden_depth + 1):
    nf = getHiddenNumFeatures(i, hidden_depth, rng, cfg, slippage=slippage)
    cfg0['hidden'][i] = {'num_features': nf}
    W = tf.Variable(tf.truncated_normal([nf0, nf],
                                        stddev = 0.01,
                                        seed=fseed))    
    b = tf.Variable(tf.constant(0.01, shape=[nf]))
    hidden = tf.nn.relu(tf.matmul(hidden, W) + b)
    print('hidden layer %d %s' % (i, str(hidden.get_shape().as_list())))
    nf0 = nf

  #decode
  ds = encode_nodes_future[encode_depth].get_shape().as_list()[1]
  nf1 = getDecodeNumFilters(0, encode_depth, rng, cfg, slippage=slippage)
  cfg0['decode'] = {0: {'num_filters': nf1}}
  if ds * ds * nf1 != nf0:
    W = tf.Variable(tf.truncated_normal([nf0, ds * ds * nf1],
                                        stddev = 0.01,
                                        seed=fseed))
    b = tf.Variable(tf.constant(0.01, shape=[ds * ds * nf1]))
    hidden = tf.matmul(hidden, W) + b
    print("Linear from %d to %d for input size %d" % (nf0, ds * ds * nf1, ds))
  decode = tf.reshape(hidden, [enc_shape[0], ds, ds, nf1])  
  print("Unflattening to", decode.get_shape().as_list())
  
  pred = tf.concat(3, [decode, encode_nodes_current[encode_depth]])
  nf = encode_nodes_future[encode_depth].get_shape().as_list()[-1]
  cfs = getDecodeFilterSize2(0, encode_depth, rng, cfg, slippage=slippage)
  W = tf.Variable(tf.truncated_normal([cfs, cfs, nf + nf1, nf],
                                      stddev=0.1,
                                      seed=fseed))
  b = tf.Variable(tf.zeros([nf]))
  pred = tf.nn.conv2d(pred,
                      W,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  pred = tf.nn.relu(tf.nn.bias_add(pred, b))
  
  norm = (ds**2) * enc_shape[0] * nf
  loss = tf.nn.l2_loss(pred - encode_nodes_future[encode_depth]) / norm
  
  for i in range(1, encode_depth + 1):
    nf0 = nf1
    ds = encode_nodes_future[encode_depth - i].get_shape().as_list()[1]
    decode = tf.image.resize_images(decode, ds, ds)
    print('Decode resize %d to shape' % i, decode.get_shape().as_list())
    cfs = getDecodeFilterSize(i, encode_depth, rng, cfg, slippage=slippage)
    cfg0['decode'][i] = {'filter_size': cfs}
    nf1 = getDecodeNumFilters(i, encode_depth, rng, cfg, slippage=slippage)
    cfg0['decode'][i]['num_filters'] = nf1
    W = tf.Variable(tf.truncated_normal([cfs, cfs, nf0, nf1],
                                        stddev=0.1,
                                        seed=fseed))
    b = tf.Variable(tf.zeros([nf1]))
    decode = tf.nn.conv2d(decode,
                          W,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    decode = tf.nn.relu(tf.nn.bias_add(decode, b))
    
    pred = tf.concat(3, [decode, encode_nodes_current[encode_depth - i]])
    
    cfs = getDecodeFilterSize2(i, encode_depth, rng, cfg, slippage=slippage)
    cfg0['decode'][i]['filter_size2'] = cfs
  
    nf = encode_nodes_future[encode_depth - i].get_shape().as_list()[-1]
    W = tf.Variable(tf.truncated_normal([cfs, cfs, nf + nf1, nf],
                                        stddev=0.1,
                                        seed=fseed))
    b = tf.Variable(tf.zeros([nf]))
    pred = tf.nn.conv2d(pred,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    pred = tf.nn.bias_add(pred, b)

    if i == encode_depth:  #add relu to all but last ... need this?
      pred = tf.minimum(tf.maximum(pred, -1), 1)
    else:
      pred = tf.nn.relu(pred)
    
    norm = (ds**2) * enc_shape[0] * nf
    loss = loss + tf.nn.l2_loss(pred - encode_nodes_future[encode_depth - i]) / norm
  #loss = loss 
 
  return loss, pred, cfg0

def loss_per_case_fn(labels, logits, **kwargs):
  #Changed names of inputs to make compatible with tfutils, but this isn't so natural...
  outputs = logits
  inputs = labels
  print('printing in and out')
  print(inputs)
  print(outputs)
  encode_depth = len(outputs['pred']) - 1
  batch_size = outputs['pred']['pred0'].get_shape().as_list()[0]
  #this just to avoid declaring another placeholder
  tv = outputs['future']['future' + str(0)]
  pred = outputs['pred']['pred' + str(0)]
  my_shape = tv.get_shape().as_list()
  norm = (my_shape[1]**2) * my_shape[0] * my_shape[-1]
  loss = tf.nn.l2_loss(pred - tv) / norm
  for i in range(1, encode_depth + 1):
    tv = outputs['future']['future' + str(i)]
    pred = outputs['pred']['pred' + str(i)]
    my_shape = tv.get_shape().as_list()
    norm = (my_shape[1]**2) * my_shape[0] * my_shape[-1]
    loss = loss + tf.nn.l2_loss(pred - tv) / norm
  return loss

def loss_agg_for_validation(labels, logits, **kwargs):
  #kind of a hack, just getting a validation score like our loss for this test
  return {'minibatch_loss' : tf.reduce_mean(loss_per_case_fn(labels, logits, **kwargs))}



def get_model(rng, batch_size, cfg, slippage, slippage_error,
              host, port, datapath, keyname,
              loss_multiple=1, diff_gated=False, diff_diff=0.1, diff_power=None):
  global sock
  if sock is None:
    initialize(host, port, datapath, keyname)

  current_node = tf.placeholder(
      tf.float32,
      shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  future_node = tf.placeholder(
        tf.float32,
      shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  actions_node = tf.placeholder(tf.float32,
                                shape=(batch_size,
                                       ACTION_LENGTH))
  
  time_node = tf.placeholder(tf.float32,
                             shape=(batch_size, 1))

  loss, train_prediction, cfg = model(current_node, future_node, 
                                      actions_node, time_node, 
                                      rng=rng, cfg=cfg, 
                                      slippage=slippage, 
                                      slippage_error=slippage_error)

  innodedict = {'current': current_node,
                'future': future_node,
                'actions': actions_node,
                'timediff': time_node}

  outnodedict = {'train_prediction': train_prediction,
                 'loss': loss}
                
  return outnodedict, innodedict, cfg  
