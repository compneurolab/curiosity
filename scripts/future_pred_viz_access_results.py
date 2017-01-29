'''
A simple script for saving viz results somewhere so we can pull them down and look at our leisure.
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
exp_id = 'test19_time10'
save_loc = '/home/nhaber/really_temp'
save_fn = os.path.join(save_loc, exp_id + '.p')
example_step = 1

conn = pm.MongoClient(port = 27017)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
cur = coll.find({'exp_id' : exp_id})


q = {'exp_id': exp_id, 'validation_results.valid0.intermediate_steps': {'$exists': True}}
res = coll.find(q)
num_valids = res.count()

print('num_valids')
print(num_valids)
exp_temp_dir = os.path.join(save_loc, exp_id)
if not os.path.exists(exp_temp_dir):
	os.mkdir(exp_temp_dir)
for i in range(num_valids):
	valid_num_dir = os.path.join(exp_temp_dir, 'valid_num_' + str(i))
	if not os.path.exists(valid_num_dir):
		os.mkdir(valid_num_dir)
	r = res[i]['validation_results']['valid0']['intermediate_steps']
	print len(r)
	for (step_num, idval) in enumerate(r):
		step_num_dir = os.path.join(valid_num_dir, 'step_' + str(step_num))
		if not os.path.exists(step_num_dir):
			os.mkdir(step_num_dir)
		# print('how big is this %d' % coll.find({'item_for' : idval}).count())
		fn = coll.find({'item_for' : idval})[0]['filename']
		fs = gridfs.GridFS(coll.database, collname)
		fh = fs.get_last_version(fn)
		saved_data = cPickle.loads(fh.read())['validation_results']['valid0']
		fh.close()
		array_data = dict((k, [np.array(elt) for elt in v]) for k, v in saved_data.iteritems())
		num_im_per_step = len(array_data[array_data.keys()[0]])
		for im_num in range(num_im_per_step):
			im_num_dir = os.path.join(step_num_dir, 'im_' + str(im_num))
			if not os.path.exists(im_num_dir):
				os.mkdir(im_num_dir)
			for k in array_data:
				if k == 'actions':
					continue
				im_fn = os.path.join(im_num_dir, k + '.jpeg')
				try:
					im = Image.fromarray(array_data[k][im_num])
				except:
					print('Failed, array type ')
					print(array_data[k][im_num])
				im.save(im_fn)

# def pickle_a_validation_num(valid_num):






# r = coll.find(q)[0]
# q1 = {'exp_id': 'validation1', 'validation_results.valid1.intermediate_steps': {'$exists': False}}


# idval = r['validation_results']['valid0']['intermediate_steps'][example_step]
# fn = coll.find({'item_for': idval})[0]['filename']
# fs = gridfs.GridFS(coll.database, collname)
# fh = fs.get_last_version(fn)
# saved_data = cPickle.loads(fh.read())
# print(saved_data['validation_results']['valid0'].keys())
# fh.close()
# saved_as_arrays = dict((k, [np.array(elt) for elt in v]) for k, v in saved_data['validation_results']['valid0'].iteritems())

# sample_image = saved_as_arrays['current_images'][0]
# im = Image.fromarray(sample_image)
# im.save(os.path.join(save_loc, 'sample.jpeg'))

# end_i = None
# has_val = []
# has_substance = []
# i = 0
# while end_i is None:
# 	print('not broken out %d' % i)
# 	try:
# 		if 'validation_results' in cur[i].keys():
# 			has_val.append(i)
# 			if len(cur[i]['validation_results']['valid0']) > 1:
# 				has_substance.append(i)
# 		i+=1
# 	except:
# 		end_i = i

# print(has_substance)
# currents = []
# futures = []
# actions = []
# preds = []
# futures_through = []
# for i in has_substance:
# 	res = cur[i]['validation_results']['valid0']
# 	currents.append(res['current_images'])
# 	futures.append(res['future_images'])
# 	actions.append(res['actions'])
# 	preds.append(res['prediction'])
# 	futures_through.append(res['futures_through'])

# to_pickle = {'currents' : currents, 'futures' : futures, 'actions' : actions, 'preds' : preds, 'futures_through' : futures_through}
# with open(save_fn, 'w') as stream:
# 	pickle.dump(to_pickle, stream)