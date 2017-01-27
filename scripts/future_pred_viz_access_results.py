'''
A simple script for saving viz results somewhere so we can pull them down and look at our leisure.
'''

import pymongo as pm
import pickle
import os

dbname = 'future_pred_test'
collname = 'symmetric_viz'
port = 27017
exp_id = 'save_im_5'
save_loc = '/home/nhaber/really_temp'
save_fn = os.path.join(save_loc, exp_id + '.p')

conn = pm.MongoClient(port = 27017)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
cur = coll.find({'exp_id' : exp_id})


end_i = None
has_val = []
has_substance = []
i = 0
while end_i is None:
	print('not broken out %d' % i)
	try:
		if 'validation_results' in cur[i].keys():
			has_val.append(i)
			if len(cur[i]['validation_results']['valid0']) > 1:
				has_substance.append(i)
		i+=1
	except:
		end_i = i

print(has_substance)
currents = []
futures = []
actions = []
preds = []
futures_through = []
for i in has_substance:
	res = cur[i]['validation_results']['valid0']
	currents.append(res['current_images'])
	futures.append(res['future_images'])
	actions.append(res['actions'])
	preds.append(res['prediction'])
	futures_through.append(res['futures_through'])

to_pickle = {'currents' : currents, 'futures' : futures, 'actions' : actions, 'preds' : preds, 'futures_through' : futures_through}
with open(save_fn, 'w') as stream:
	pickle.dump(to_pickle, stream)