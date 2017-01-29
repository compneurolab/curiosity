'''
A little script to access db.
'''

import pymongo as pm

dbname = 'future_pred_test'
collname = 'future_pred_symmetric'
port = 27017
exp_id = 'test12_sepval'

conn = pm.MongoClient(port = 27017)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
cur = coll.find({'exp_id' : exp_id})

end_i = None
has_val = []
has_train = []
i = 0
while end_i is None:
	print('not broken out %d' % i)
	try:
		if 'validation_results' in cur[i].keys():
			has_val.append(i)
		if 'train_results' in cur[i].keys():
			has_train.append(i)
		i+=1
	except:
		end_i = i

print('duration')

for i in range(end_i):
	print(cur[i]['duration'])

print('validation')

for i in has_val:
	print(cur[i]['validation_results'])

print('summary training loss')

for i in has_train:
	print(cur[i]['train_results'][-1])






