import h5py
import numpy
from PIL import Image
import json
import os

BIG_PATH = '/media/data3/new_dataset'
TEST_SAVE_DIR = os.path.join(BIG_PATH, 'test_images')
BATCH_SIZE = 256

f = h5py.File(os.path.join(BIG_PATH, 'new_dataset.hdf5'), mode = 'r')

def print_actions(begin_bn, end_bn):
	actions = [json.loads(act) for act in f['actions'][begin_bn * BATCH_SIZE : end_bn * BATCH_SIZE]]
	for (i, act) in enumerate(actions):
		print(i)
		print(act)

def print_num_tables_not_tables(begin_bn, end_bn):
	infos = [json.loads(info) for info in f['worldinfo'][begin_bn * BATCH_SIZE : end_bn * BATCH_SIZE]]
	for info in infos:
		obs_obj = info['observed_objects']
		tables = [o for o in obs_obj if o[5] and int(o[1]) != -1 and not o[4]]
		not_tables = [o for o in obs_obj if not o[5] and int(o[1]) != -1 and not o[4]]
		print((len(tables), len(not_tables)))

def write_images(begin_bn, end_bn, dir_name):
	assert BATCH_SIZE * (end_bn - begin_bn) < 10001
	full_dir_name = os.path.join(TEST_SAVE_DIR, dir_name)
	if os.path.isdir(full_dir_name):
		raise Exception('Directory already exists!')
	else:
		os.mkdir(full_dir_name)
	descs = ['images', 'objects', 'normals', 'images2', 'objects2', 'normals2']
	for desc in descs:
		print(desc)
		imagestuffs = f[desc][begin_bn * BATCH_SIZE : end_bn * BATCH_SIZE]
		for (i, img) in enumerate(imagestuffs):
			print(i)
			counter = str(i)
			while len(counter) < 4:
				counter = '0' + counter
			fn = os.path.join(full_dir_name, desc + '_' + counter + '.png')
			im = Image.fromarray(img)
			im.save(fn)

def get_num_valid_batches():
	valid = f['valid']
	for i in range(0, valid.shape[0], BATCH_SIZE):
		if not valid[i : i + BATCH_SIZE].all():
			return int(i/BATCH_SIZE)
	return int(valid.shape[0])

print get_num_valid_batches()
		
#write_images(13, 14, 'lucky_13')


f.close()


