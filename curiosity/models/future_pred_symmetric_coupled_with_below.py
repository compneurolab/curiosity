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
  new_inputs = {'current' : inputs['images'], 'actions' : inputs['parsed_actions'], 'future' : inputs['future_images'], 'time' : tf.ones([batch_size, 1])}
  return model_tfutils(new_inputs, **kwargs)


def model_tfutils(inputs, rng, cfg = {}, train = True, slippage = 0, diff_mode = False, num_classes = 1, **kwargs):
  '''Model definition, compatible with tfutils.

  inputs should have 'current', 'future', 'action', 'time' keys. Outputs is a dict with keys, pred and future, within those, dicts with keys predi and futurei for i in 0:encode_depth, to be matched up in loss.
  num_classes = 1 is equivalent to the original l2 loss model.
  '''
  current_node = inputs['current']
  future_node = inputs['future']
  actions_node = inputs['actions']
  time_node = inputs['time']

  current_node = tf.divide(tf.cast(current_node, tf.float32), 255)
  future_node = tf.divide(tf.cast(future_node, tf.float32), 255)
  actions_node = tf.cast(actions_node, tf.float32)
  print('Actions shape')
  print(actions_node.get_shape().as_list())


  print('Diff mode: ' + str(diff_mode))

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
        print('Current encode node shape: ' + str(new_encode_node_current.get_shape().as_list()))
    with tf.variable_scope('encode' + str(i), reuse = True):
      new_encode_node_future = m.conv(nf, cfs, cs, in_layer = encode_nodes_future[i - 1], \
        init='trunc_norm', stddev=.01, bias=0, activation='relu')
  #TODO add print function
      print('Future encode node shape: ' + str(new_encode_node_current.get_shape().as_list()))
      do_pool = getEncodeDoPool(i, encode_depth, rng, cfg, slippage=slippage)
      if do_pool:
        pfs = getEncodePoolFilterSize(i, encode_depth, rng, cfg, slippage=slippage)
        ps = getEncodePoolStride(i, encode_depth, rng, cfg, slippage=slippage)
        pool_type = getEncodePoolType(i, encode_depth, rng, cfg, slippage=slippage)
        print('Pool size %d, stride %d' % (pfs, ps))
        print('Type: ' + pool_type)
        #just correcting potential discrepancy in descriptor
        if pool_type == 'max':
          pool_type = 'maxpool'
        new_encode_node_current = m.pool(pfs, ps, in_layer = new_encode_node_current, pfunc = pool_type)
        new_encode_node_future = m.pool(pfs, ps, in_layer = new_encode_node_future, pfunc = pool_type)
        print('Current encode node shape: ' + str(new_encode_node_current.get_shape().as_list()))
        print('Future encode node shape: ' + str(new_encode_node_future.get_shape().as_list()))        
      encode_nodes_current.append(new_encode_node_current)
      encode_nodes_future.append(new_encode_node_future)

  with tf.variable_scope('addactiontime'):
    encode_node = encode_nodes_current[-1]
    enc_shape = encode_node.get_shape().as_list()
    encode_flat = m.reshape([np.prod(enc_shape[1:])], in_layer = encode_node)
    print('Flatten to shape %s' % encode_flat.get_shape().as_list())
    if time_node is not None:
      encode_flat = m.add_bypass([actions_node, time_node])
    else:
      encode_flat = m.add_bypass(actions_node)

  nf0 = encode_flat.get_shape().as_list()[1]
  hidden_depth = getHiddenDepth(rng, cfg, slippage=slippage)
  print('Hidden depth: %d' % hidden_depth)
  hidden = encode_flat
  for i in range(1, hidden_depth + 1):
    with tf.variable_scope('hidden' + str(i)):
      nf = getHiddenNumFeatures(i, hidden_depth, rng, cfg, slippage=slippage)
      #TODO: this can be made nicer once we add more general concat
      hidden = m.fc(nf, init = 'trunc_norm', activation = 'relu', bias = .01, in_layer = hidden, dropout = None)
      print('Hidden shape %s' % hidden.get_shape().as_list())
      nf0 = nf


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
      print('Shape after bypass %s' % pred.get_shape().as_list())
      nf = encode_nodes_future[encode_depth - i].get_shape().as_list()[-1]
      cfs = getDecodeFilterSize2(i, encode_depth, rng, cfg, slippage = slippage)
      print('Pred conv filter size %d' % cfs)
      if i == encode_depth:
        pred = m.conv(nf * num_classes, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation=None)
        #making this another dimension, while I *think* conv_2d would not handle this
        if num_classes > 1:
          my_shape = pred.get_shape().as_list()
          my_shape[3] = nf
          my_shape.append(num_classes)
          pred = m.reshape(my_shape[1:])
      else:
        if diff_mode:
          pred = m.conv(nf, cfs, 1, init='trunc_norm', stddev=.1, bias=0, activation=None)
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
        print('Decode conv to shape %s' % decode.get_shape().as_list())

  enc_string = None
  enc_dict = None
  if diff_mode:
    diffs = [encoded_future - encoded_current for (encoded_current, encoded_future) in zip(encode_nodes_current, encode_nodes_future)]
    encode_nodes_diff_dict = dict(('diff' + str(i), diff) for (i, diff) in enumerate(diffs))
    enc_string = 'diff'
    enc_dict = encode_nodes_diff_dict
  else:
    encode_nodes_future_dict = dict(('future' + str(i), encoded_future) for (i, encoded_future) in enumerate(encode_nodes_future))
    enc_string = 'future'
    enc_dict = encode_nodes_future_dict
  outputs = {'pred' : preds, enc_string: enc_dict}


  return outputs, m.params







def diff_loss_per_case_fn(labels, logits, **kwargs):
  '''This allows us to do the diff one while reusing the above code.

  Maybe merge with below.'''
  #Changed names of inputs to make compatible with tfutils, but this isn't so natural...
  outputs = logits
  inputs = labels
  encode_depth = len(outputs['pred']) - 1
  batch_size = outputs['pred']['pred0'].get_shape().as_list()[0]
  #this just to avoid declaring another placeholder
  tv = outputs['diff']['diff' + str(0)]
  pred = outputs['pred']['pred' + str(0)]
  my_shape = tv.get_shape().as_list()
  norm = (my_shape[1]**2) * my_shape[0] * my_shape[-1]
  loss = tf.nn.l2_loss(pred - tv) / norm
  for i in range(1, encode_depth + 1):
    tv = outputs['diff']['diff' + str(i)]
    pred = outputs['pred']['pred' + str(i)]
    my_shape = tv.get_shape().as_list()
    norm = (my_shape[1]**2) * my_shape[0] * my_shape[-1]
    loss = loss + tf.nn.l2_loss(pred - tv) / norm
  return loss


def loss_per_case_fn(labels, logits, **kwargs):
  #Changed names of inputs to make compatible with tfutils, but this isn't so natural...
  outputs = logits
  inputs = labels
  encode_depth = len(outputs['pred']) - 1
  batch_size = outputs['pred']['pred0'].get_shape().as_list()[0]
  #this just to avoid declaring another tensor
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

def discretized_loss_fn(labels, logits, num_classes, sigmoid_hiddens = False, **kwargs):
  outputs = logits
  inputs = labels
  encode_depth = len(outputs['pred']) - 1
  tv = outputs['diff']['diff0']
  tv = tf.cast((num_classes - 1) * tv, tf.uint8)
  tv = tf.one_hot(tv, depth = num_classes)
  pred = outputs['pred']['pred0']
  #Not sure whether we should normalize this at all, but I think it's pretty ok as is
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, tv))
  for i in range(1, encode_depth + 1):
    tv = outputs['diff']['diff' + str(i)]
    pred = outputs['pred']['pred' + str(i)]
    if sigmoid_hiddens:
      pred = 2 * tf.nn.sigmoid(pred) - 1
    my_shape = tv.get_shape().as_list()
    norm = (my_shape[1]**2) * my_shape[0] * my_shape[-1]
    loss = loss + tf.nn.l2_loss(pred - tv) / norm
  return loss

def something_or_nothing_loss_fn(labels, logits, sigmoid_hiddens = False, **kwargs):
  outputs = logits
  inputs = labels
  encode_depth = len(outputs['pred']) - 1
  #we set num_classes = 1 for this, keeping parameters down...this is probably not that important
  tv = outputs['diff']['diff0']
  tv = tf.cast(tf.ceil(tv), 'uint8')
  tv = tf.one_hot(tv, depth = 2)
  pred = outputs['pred']['pred0']
  my_shape = pred.get_shape().as_list()
  my_shape.append(1)
  pred = tf.reshape(pred, my_shape)
  pred = tf.concat(4, [tf.zeros(my_shape), pred])
  print('before loss shapes')
  print(pred.get_shape().as_list())
  print(tv.get_shape().as_list())
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, tv))
  for i in range(1, encode_depth + 1):
    tv = outputs['diff']['diff' + str(i)]
    pred = outputs['pred']['pred' + str(i)]
    if sigmoid_hiddens:
      pred = 2 * tf.nn.sigmoid(pred) - 1
    my_shape = tv.get_shape().as_list()
    norm = (my_shape[1]**2) * my_shape[0] * my_shape[-1]
    loss = loss + tf.nn.l2_loss(pred - tv) / norm
  return loss

def loss_agg_for_validation(labels, logits, **kwargs):
  #kind of a hack, just getting a validation score like our loss for this test
  return {'minibatch_loss' : tf.reduce_mean(loss_per_case_fn(labels, logits, **kwargs))}



