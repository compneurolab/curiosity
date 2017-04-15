'''
While I'm having trouble with jupyter...
'''

import pymongo as pm
import gridfs
import cPickle

dbname = 'future_prediction'
collname = 'choice_2'
port = 27017
exp_id = 'test1'
save_loc = '/home/nhaber/temp'
target_name = 'valid0'


conn = pm.MongoClient(port = port)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
# cur = coll.find({'exp_id' : exp_id})


q = {'exp_id': exp_id, 'validation_results': {'$exists': True}}

val_steps = coll.find(q)

val_num = 1
idx = val_steps[val_num]['_id']
fn = coll.find({'item_for' : idx})[0]['filename']
fs = gridfs.GridFS(coll.database, collname)
fh = fs.get_last_version(fn)
saved_data = cPickle.loads(fh.read())['validation_results']
fh.close()


