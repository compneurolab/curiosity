'''
A script for accessing visualization data (saving images at validation steps during training) and saving them to a local directory.
'''
import pymongo as pm
import pickle
import os
import gridfs
import cPickle
import numpy as np
from PIL import Image

dbname = 'future_pred_test'
collname = 'discretized'
port = 27017
exp_id = 'test1'
save_loc = '/home/nhaber/really_temp'
save_fn = os.path.join(save_loc, exp_id + '.p')
target_name = 'valid0'

conn = pm.MongoClient(port = 27017)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
cur = coll.find({'exp_id' : exp_id})

q = {'exp_id' : exp_id, 'validation_results' : {'$exists' : True}}
val_steps = coll.find(q)
val_count = val_steps.count()
print('num val steps so far')
print(val_count)

saved_data = {}

def convert_to_viz(np_arr):
	'''I did a silly thing and saved discretized-loss predictions as if they were image predictions.

	This recovers and converts to an ok visualization.'''
	my_shape = np_arr.shape
	num_classes = np_arr.shape[-1]
	#I fixed things so that it saves the prediction not converted to 255
	if np_arr.dtype == 'float32':
		exp_arr = np.exp(np_arr)
	else:
		exp_arr = np.exp(np_arr.astype('float32') / 255.)
	sum_arr = np.sum(exp_arr, axis = -1)
	#hack for broadcasting...I don't know broadcasting
	softy = (exp_arr.T / sum_arr.T).T
	return np.sum((softy * range(num_classes) * 255. / float(num_classes)), axis = -1).astype('uint8')

def convert_to_viz_sharp(np_arr):
	'''Similar to the above, but just taking the argmax, hopefully giving a sharper visualization.

	'''
	num_classes = np_arr.shape[-1]
	a_m = np.argmax(np_arr, axis = -1)
	return (a_m * 255. / float(num_classes)).astype('uint8')






for val_num in range(val_count):
	idx = val_steps[val_num]['_id']
	fn = coll.find({'item_for' : idx})[0]['filename']
	fs = gridfs.GridFS(coll.database, collname)
	fh = fs.get_last_version(fn)
	saved_data[val_num] = cPickle.loads(fh.read())['validation_results']
	fh.close()



exp_dir = os.path.join(save_loc, exp_id)
if not os.path.exists(exp_dir):
	os.mkdir(exp_dir)

for val_num, val_data in saved_data.iteritems():
	val_dir = os.path.join(exp_dir, 'val_' + str(val_num))
	if not os.path.exists(val_dir):
		os.mkdir(val_dir)
	for tgt_desc, tgt in val_data[target_name].iteritems():
		tgt_images = [arr for step_results in tgt for arr in step_results]
		for (instance_num, arr) in enumerate(tgt_images):
			instance_dir = os.path.join(val_dir, 'instance_' + str(instance_num))
			if not os.path.exists(instance_dir):
				os.mkdir(instance_dir)
			if len(arr.shape) == 4:
				fn = os.path.join(instance_dir, tgt_desc + '_' + str(instance_num) + '.jpeg')
				arr = convert_to_viz_sharp(arr)
				im = Image.fromarray(arr)
				im.save(fn)
			#just save in human-readable form if 1-array
			elif len(arr.shape) == 1:
				fn = os.path.join(instance_dir, tgt_desc + '_' + str(instance_num) + '.txt')
				np.savetxt(fn, arr)
			else:
				assert len(arr.shape) == 3
				fn = os.path.join(instance_dir, tgt_desc + '_' + str(instance_num) + '.jpeg')
				im = Image.fromarray(arr)
				im.save(fn)




