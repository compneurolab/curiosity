'''
A little script to access db.
'''

import pymongo as pm

dbname = 'future_pred_test'
collname = 'discretized'
port = 27017
exp_id = 'test1'

conn = pm.MongoClient(port = 27017)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
cur = coll.find({'exp_id' : exp_id})

count = cur.count()

q = {'exp_id' : exp_id, 'validation_results' : {'$exists' : True}}
val_steps = coll.find(q)

for i in range(val_steps.count()):
	err = sum(val_steps[i]['validation_results']['valid0'].values())
	if err > 100.:
		print i
		print err


# for i in range(count):
# 	if 'validation_results' in cur[i].keys():
# 		print i
# 		print cur[i]['validation_results']

# end_i = None
# has_val = []
# has_train = []
# i = 0
# while end_i is None:
# 	print('not broken out %d' % i)
# 	try:
# 		if 'validation_results' in cur[i].keys():
# 			has_val.append(i)
# 		if 'train_results' in cur[i].keys():
# 			has_train.append(i)
# 		i+=1
# 	except:
# 		end_i = i

# print('duration')

# for i in range(end_i):
# 	print(cur[i]['duration'])

# print('validation')

# for i in has_val:
# 	print(cur[i]['validation_results'])

# print('summary training loss')

# for i in has_train:
# 	print(cur[i]['train_results'][-1])






