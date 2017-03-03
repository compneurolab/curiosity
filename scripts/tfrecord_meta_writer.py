import os
import tensorflow as tf
import cPickle

data2_dir = '/media/data2/one_world_dataset/tfvaldata'
meta_loc = os.path.join(data2_dir, 'act_mask_1', 'meta.pkl')

mdat = {'act_mask_1': {'dtype': tf.string, 'shape': []}}

with open(meta_loc, 'w') as stream:
	cPickle.dump(mdat, stream)