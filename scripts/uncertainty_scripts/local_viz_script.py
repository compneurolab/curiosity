'''
Local viz script for LocalSaver results
'''

import os

load_sup_dir = '/Users/nickhaber/Desktop/'
exp_id = 'savetest10'
load_dir = os.path.join(load_sup_dir, exp_id)

num_files = 1


total_res = None

for i in range(1, num_files + 1):
	