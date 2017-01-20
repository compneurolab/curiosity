'''
A little script to access db.
'''

import pymongo as pm

dbname = 'future_pred_test'
collname = 'future_pred_symmetric'
port = 27017
exp_id = 'test6_wval'

conn = pm.MongoClient(port = 27017)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
cur = coll.find({'exp_id' : exp_id})