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
collname = 'future_pred_symmetric'
port = 27017
exp_id = '21_t4'
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
			#just save in human-readable form if not a 3-array
			if len(arr.shape) != 3:
				fn = os.path.join(instance_dir, tgt_desc + '_' + str(instance_num) + '.txt')
				np.savetxt(fn, arr)
			else:
				fn = os.path.join(instance_dir, tgt_desc + '_' + str(instance_num) + '.jpeg')
				im = Image.fromarray(arr)
				im.save(fn)




