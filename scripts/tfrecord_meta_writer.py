import os
import tensorflow as tf
import cPickle

data2_dir = '/media/data2/one_world_dataset/tfvaldata'
attr_name = 'pos_squeezed'
meta_loc = os.path.join(data2_dir, attr_name, 'meta.pkl')

mdat = {attr_name: {'dtype': tf.string, 'shape': []}}

with open(meta_loc, 'w') as stream:
	cPickle.dump(mdat, stream)