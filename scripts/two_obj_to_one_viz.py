'''
While I'm having trouble with jupyter...
'''

import pymongo as pm
import gridfs
import cPickle
import os
import numpy as np


DUMP_DIR = '/mnt/fs0/nhaber/pickles'
dbname = 'future_prediction'
collname = 'choice_2'
port = 27017
exp_id = 'cfg_simple_norm'
save_loc = '/home/nhaber/temp'
target_name = 'valid0'


conn = pm.MongoClient(port = port)
coll = conn[dbname][collname + '.files']
print('experiments')
print(coll.distinct('exp_id'))
# cur = coll.find({'exp_id' : exp_id})

q_all = {'exp_id' : exp_id}
q = {'exp_id': exp_id, 'validation_results': {'$exists': True}}
q_train = {'exp_id' : exp_id, 'train_results' : {'$exists' : True}}

# val_steps = coll.find(q)

# val_num = 1
# idx = val_steps[val_num]['_id']
# fn = coll.find({'item_for' : idx})[0]['filename']
# fs = gridfs.GridFS(coll.database, collname)
# fh = fs.get_last_version(fn)
# saved_data = cPickle.loads(fh.read())['validation_results']
# fh.close()


train_res = coll.find(q_train)
train_stuff = [res for i in range(train_res.count()) for res in train_res[i]['train_results']]
train_loss = np.array([res['loss'] for res in train_stuff])
train_lr = np.array([res['learning_rate'] for res in train_stuff])

with open(os.path.join(DUMP_DIR, exp_id + 'loss.p'), 'w') as stream:
	cPickle.dump(train_loss, stream)
