import numpy as np

'''
This file contains functions to read out the configuration data 
from the config file or to determine parameters randomly within 
a reasonable scope of parameter choices
'''

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
    if val is 'max':
      val = 'maxpool'
    return val
  val = rng.choice(['max', 'avg'])
  if val is 'max':
    val = 'maxpool'
  return val

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

def getEncodeParam(i, encode_depth, rng, cfg, prev=None, slippage=0):
  cfs = getEncodeConvFilterSize(i, encode_depth, rng, cfg, prev=prev, slippage=slippage)
  nf = getEncodeConvNumFilters(i, encode_depth, rng, cfg, slippage=slippage)
  cs = getEncodeConvStride(i, encode_depth, rng, cfg, slippage=slippage)

  do_pool = getEncodeDoPool(i, encode_depth, rng, cfg, slippage=slippage)
  pfs = getEncodePoolFilterSize(i, encode_depth, rng, cfg, slippage=slippage)
  ps = getEncodePoolStride(i, encode_depth, rng, cfg, slippage=slippage)
  pool_type = getEncodePoolType(i, encode_depth, rng, cfg, slippage=slippage)
  
  return[cfs, nf, cs, do_pool, pfs, ps, pool_type]
