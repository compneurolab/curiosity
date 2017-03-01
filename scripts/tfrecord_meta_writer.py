import os
import tensorflow as tf
import cPickle

data2_dir = '/media/data2/one_world_dataset/tfdata'
meta_loc = os.path.join(data2_dir, 'positions', 'meta.pkl')

mdat = {'positions': {'dtype': tf.string, 'shape': []}}

with open(meta_loc, 'w') as stream:
	cPickle.dump(mdat, stream)