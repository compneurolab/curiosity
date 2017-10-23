import h5py
import cPickle
import numpy as np
import os
import sys
import json
from tqdm import trange

# open file
f = h5py.File(sys.argv[1], 'r+')
assert 'msg' in f
N = len(f['msg'])

# new data is object centroid if object is there otherwise (-1, -1) and object is there
hdf5_handles = {
        'pos': f.require_dataset('pos', shape = (N,2), \
                dtype = h5py.special_dtype(vlen = str)),
        'is_there': f.require_dataset('is_there', shape = (N,), \
                dtype = h5py.special_dtype(vlen = str)),
        }
# find centroid and if object is there and write to file
for i in trange(N-1):
    msg = f['msg'][i+1]
    objects = f['objects1'][i]
    pos = (-1, -1)
    is_there = 0
    if msg != 'null':
        msg = json.loads(msg)
        if msg['msg']['action_type'] == 'OBJ_ACT':
            objects = 256**2*objects[:, :, 0] + 256*objects[:, :, 1] + objects[:, :, 2]
            object_id = int(msg['msg']['actions'][0]['id'])
            xs, ys = (objects == object_id).nonzero()
            pos = list(zip(xs, ys))
            if len(xs) > 0:
                pos = np.array(pos).mean(0)
                is_there = 1
            else:
                raise IndexError('Impossible action - objects combination!')
    hdf5_handles['pos'][i] = pos
    hdf5_handles['is_there'][i] = is_there
hdf5_handles['pos'][N-1] = (-1, -1)
hdf5_handles['is_there'][N-1] = 0
f.close()
