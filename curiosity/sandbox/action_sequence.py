from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from StringIO import StringIO
import math
import sys
import copy
import numpy as np
import time
import os
import zmq
import struct
import json
from PIL import Image

import gzip

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 256
ENCODE_DIMS = 1024
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
SEED = 0  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPISODES = 1000000
EPISODE_LENGTH = 50
OBSERVATION_LENGTH = 2
ENCODE_DEPTH = 1
PHYSNET_DEPTH = 2
DECODE_DEPTH = 1
MAX_NUM_ACTIONS = 5
ATOMIC_ACTION_LENGTH = 14

rng = np.random.RandomState(0)

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

if 1:
  ctx = zmq.Context()
  sock = ctx.socket(zmq.REQ)

  print("connecting...")
  sock.connect("tcp://18.93.15.188:23042")
  print("...connected")
  sock.send(json.dumps({'n': 4, 'msg': {"msg_type": "CLIENT_JOIN"}}))
  print("...joined")


def getEpisode():
  ims = []
  norms = []
  actions = []
  timestep = 0
  first = True 
  while timestep < EPISODE_LENGTH + 1:
    info, nstr, ostr, imstr = handle_message(sock)
    objarray = np.asarray(Image.open(StringIO(ostr)).convert('RGB'))
    normalsarray = np.asarray(Image.open(StringIO(nstr)).convert('RGB'))
    imarray = np.asarray(Image.open(StringIO(imstr)).convert('RGB'))
    msg = {'n': 4,
           'msg': {"msg_type": "CLIENT_INPUT",
                   "actions": [],
                   "get_obj_data": False}}

    objarray = 256**2 * objarray[:, :, 0] + 256 * objarray[:, :, 1] + objarray[:, :, 2]
    objs = np.unique(objarray) 
    objs = objs[objs > 2] 
    if first or len(objs) == 0:
      print('teleporting at %d ... ' % timestep)
      msg['msg']['teleport_random'] = True
    else:
      x, y = choose_action_position(objarray)
      o = objarray[x, y]
      if timestep < MAX_NUM_ACTIONS:
        msg['msg']['actions'] = [{'id': str(o),
                                  'action_pos': [x, y],
                                  'force': [rng.choice([-10, 0, 10]),
                                            20,
                                            rng.choice([-10, 0, 10])],
                                  'torque': [0, 0, 0]}]
      #every few frames, shift around a little 
      if timestep % 5 == 0:
        msg['msg']['vel'] = [.3 * rng.uniform(), 0.15 * rng.uniform(), 0.3 * rng.uniform()]
      
      timestep += 1
           
      ims.append(imarray)
      norms.append(normalsarray)
      actions.append(copy.deepcopy(msg['msg']))
    first = False
    sock.send_json(msg)

  def norml(x):
    return (x - PIXEL_DEPTH/2.0) / PIXEL_DEPTH
  ims = norml(np.array(ims))
  norms = norml(np.array(norms))
   
  batch = {'images': norml(np.array(ims)),
           'normals': norml(np.array(norms)),
           'actions': actions}
  return batch


def process_action(action):
   act = action.get('actions', [{'action_pos': [0, 0], 'force': [0, 0, 0], 'torque': [0, 0, 0]}])[0]
   return [action.get('vel', 0), action.get('ang_vel', 0)] + act['action_pos'] + act['force'] + a['torque']


def getNextBatch(N):
  batch = getEpisode()
  normals = batch['normals']
  actions = batch['actions']
  k = len(batch['normals'])
  obss = []
  futures = []
  time_diffs = []
  actions = []
  for i in range(N):
    j0 = rng.randint(k - OBSERVATION_LENGTH)
    j1 = rng.randint(low = j0 + OBSERVATION_LENGTH, high=k)
    newshape = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS * OBSERVATION_LENGTH)
    obs = normals[j0: j0 + OBSERVATION_LENGTH].transpose((1, 2, 3, 0)).reshape(newshape)
    obs.append(obs)
    future_inds.append(j1)
    time_diffs.append(j1 - j0)
    action_seq = itertools.chain(*[process_actions(actions[_j]) for _j in range(j0, j0 + OBSERVATION_LENGTH)])
    actions.append(action_seq)

  batch = {'observations': np.array(obss),
            'future': normals[future_inds],
            'actions': np.array(actions),
            'time_diffs': np.array(time_diffs)}
            
  return batch


def error_rate(predictions, imgs):
  """Return the error rate based on dense predictions and sparse labels."""
  return 0.5 * ((predictions - imgs)**2).mean()


def getDecodeNumFilters(i, N):
  if i < N:
     return 4
  else:
     return NUM_CHANNELS

def getDecodeFilterSize(i, N):
  return 7

def getDecodeSizes(N, initial_size, final_size):
  s = np.log2(initial_size)
  e = np.log2(final_size)
  increment = (e - s) / N
  l = np.around(np.power(2, np.arange(s, e, increment)))
  if len(l) < N + 1:
    l = np.concatenate([l, [final_size]])
  l = l.astype(np.int) 
  return l
  
def getPhysnetNumFilters(i, N):
  return 4

def getEncodeNumFilters(i, N):
  return 4
  L = [64, 128, 256, 512, 1024, 512]
  return L[i] if i < len(L) else 256

def getEncodeFilterSize(i, N):
  L = [7, 5]
  return L[i] if i < len(L) else 3

def getEncodeConvStride(i, N):
  return 1

def getEncodePoolFilterSize(i, N):
  return 2

def getEncodePoolStride(i, N):
  return 2


def main(argv):
  #holder for observation data
  observation_node = tf.placeholder(tf.float32,
                                    shape=(BATCH_SIZE,
                                           IMAGE_SIZE,
                                           IMAGE_SIZE,
                                           NUM_CHANNELS * OBSERVATION_LENGTH))
  #holder for prediction  
  future_node = tf.placeholder(tf.float32,
                               shape=(BATCH_SIZE,
                                      IMAGE_SIZE,
                                      IMAGE_SIZE,
                                      NUM_CHANNELS))
  
  #holder for action space element
  actions_node = tf.placeholder(tf.float32,
                                shape=(BATCH_SIZE, 
                                       ATOMIC_ACTION_LENGTH * MAX_NUM_ACTIONS))
 
  #time forward
  time_node = tf.placeholder(tf.float32,
                             shape=(BATCH_SIZE, 1))


  def model(obs_node, actions_node, time_node):
    """The Model definition."""

    # encoding phase for image-shaped observations == could be empty
    nf0 = NUM_CHANNELS * OBSERVATION_LENGTH
    imsize = IMAGE_SIZE
    for i in range(1, ENCODE_DEPTH + 1):
      cfs = getEncodeFilterSize(i, ENCODE_DEPTH)
      nf = getEncodeNumFilters(i, ENCODE_DEPTH)
      cs = getEncodeConvStride(i, ENCODE_DEPTH)
      pfs = getEncodePoolFilterSize(i, ENCODE_DEPTH)
      ps = getEncodePoolStride(i, ENCODE_DEPTH)
      W = tf.Variable(tf.truncated_normal([cfs, cfs, nf0, nf],
                                           stddev=0.01,
                                           seed=SEED))
      b = tf.Variable(tf.zeros([nf]))
      obs_node = tf.nn.relu(tf.nn.conv2d(obs_node, 
                                         W,
                                         strides=[1, cs, cs, 1],
                                         padding='SAME'))
      obs_node = tf.nn.bias_add(obs_node, b)
      obs_node = tf.nn.max_pool(obs_node, 
                                ksize=[1, pfs, pfs, 1], 
                                strides=[1, ps, ps, 1], 
                                padding='SAME')
      nf0 = nf 
      imsize = imsize // (cs * ps)
      
    #flatten the observations
    obs_shape = obs_node.get_shape().as_list()
    obs_flat = tf.reshape(obs_node, [obs_shape[0], np.prod(obs_shape[1:])])
                    
    #concatenate223
    concat = tf.concat(1, [obs_flat, actions_node, time_node])
    #apply physics neural network: currently MLP
    #TODO: maybe should be convnet?   time channel? + action channels?  recurrence? 
    nf0 = imsize * imsize * nf0 + MAX_NUM_ACTIONS * ATOMIC_ACTION_LENGTH + 1
    for i in range(1, PHYSNET_DEPTH + 1):
      nf = getPhysnetNumFilters(i, PHYSNET_DEPTH)
      W = tf.Variable(tf.truncated_normal([nf0, nf],
                                          stddev = 0.01,
                                          seed=SEED))
      b = tf.Variable(tf.constant(0.01, shape=[nf]))
      concat = tf.nn.relu(tf.matmul(concat, W) + b)
      nf0 = nf
                    
    #decode
    #first, unflatten
    nf = getDecodeNumFilters(0, DECODE_DEPTH)
    ds0 = int(math.ceil(math.sqrt(nf0 / nf)))
    dsizes = getDecodeSizes(DECODE_DEPTH, ds0, IMAGE_SIZE)
    ds = dsizes[0]
    if ds * ds * nf != nf0:
      W = tf.Variable(tf.truncated_normal([nf0, ds * ds * nf],
                                          stddev = 0.01,
                                          seed=SEED))
      b = tf.Variable(tf.constant(0.01, shape=[nf]))
      concat = tf.matmul(concat, W) + b
    decode = tf.reshape(concat, [BATCH_SIZE, ds, ds, nf])
    for i in range(1, DECODE_DEPTH + 1):
      nf0 = nf
      ds = dsizes[i]
      if i == DECODE_DEPTH:
         assert ds == IMAGE_SIZE, (ds, IMAGE_SIZE)
      decode = tf.image.resize_images(decode, ds, ds)

      cfs = getDecodeFilterSize(i, DECODE_DEPTH)
      nf = getDecodeNumFilters(i, DECODE_DEPTH)
      if i == DECODE_DEPTH:
         assert nf == NUM_CHANNELS, (nf, NUM_CHANNELS)
      W = tf.Variable(tf.truncated_normal([cfs, cfs, nf0, nf],
                                           stddev=0.01,
                                           seed=SEED))
      b = tf.Variable(tf.zeros([nf]))
      decode = tf.nn.conv2d(decode,
                            W,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
      decode = tf.nn.bias_add(decode, b)
      if i < DECODE_DEPTH:  #add relu to all but last ... need this?
         decode = tf.nn.relu(decode)

    return decode

  train_prediction = model(observation_node, actions_node, time_node)  
  norm = (IMAGE_SIZE**2) * NUM_CHANNELS * BATCH_SIZE
  loss = tf.nn.l2_loss(train_prediction - future_node) / norm

  batch = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(
      1.,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      100000,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)

  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

  start_time = time.time()
  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print('Initialized!')
    for episode in xrange(NUM_EPISODES):
      print('there')
      batch_data = getNextBatch(BATCH_SIZE)
      print('here')
      feed_dict = {observation_node: batch_data['observations'],
                   actions_node: batch_data['actions'], 
                   time_node: batch_data['time_diff'], 
                   future_node: batch_data['future']}

      _, l, lr, predictions = sess.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      print(episode, l, lr)


def handle_message(sock, write=False, outdir='', imtype='png', prefix=''):
    t0 = time.time()
    msg = sock.recv()
    img0 = sock.recv()
    img1 = sock.recv()
    img2 = sock.recv()
    t1 = time.time()
    if write:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        with open(os.path.join(outdir, 'image_%s.%s' % (prefix, imtype)), 'w') as _f:
            _f.write(img2)
        with open(os.path.join(outdir, 'objects_%s.%s' % (prefix, imtype)), 'w') as _f:
            _f.write(img1)
        with open(os.path.join(outdir, 'normals_%s.%s' % (prefix, imtype)), 'w') as _f:
            _f.write(img0)
        with open(os.path.join(outdir, 'info_%s.json' % prefix), 'w') as _f:
            _f.write(msg)
    return [msg, img0, img1, img2]


def choose_action_position(objarray):
  xs, ys = (objarray > 2).nonzero()
  pos = zip(xs, ys)
  return pos[rng.randint(len(pos))]

if __name__ == '__main__':
  tf.app.run()
