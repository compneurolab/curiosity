import h5py
import numpy as np
import os

paths = ['/media/data2/one_world_dataset', '/media/data/two_world_dataset']
filenames = [['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset4_resized', 'dataset5', 'dataset6', 'dataset8', 'dataset9_resized', 'dataset10', 'dataset11'], ['dataset1', 'dataset2', 'dataset3', 'dataset5', 'dataset7']]
big_hdf5_filename = '/media/data3/new_dataset/new_dataset.hdf5'
other_big_filename = '/media/data3/new_dataset/new_dataset1.hdf5'

BATCH_SIZE = 256
CURR_SIZE = 70

SCREEN_HEIGHT = 256
SCREEN_WIDTH = 600

NUM_CURRICULA_TOT = 115

def get_valid_num_batches(my_hdf5):
	valid = my_hdf5['valid']
	print(valid.shape)
	num_batches_allocated = int(valid.shape[0]/BATCH_SIZE)
	for bn in range(num_batches_allocated):
		if not valid[bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE].all():
			return bn
	return num_batches_allocated

def get_counts():
	num_batches_tot = 0
	num_curricula_tot = 0
	for (path, fns) in zip(paths, filenames):
		for fn in fns:
			filename = os.path.join(path, fn + '.hdf5')
			print(filename)
			my_hdf5 = h5py.File(filename, 'r')
			num_batches = get_valid_num_batches(my_hdf5)
			num_curricula = int(num_batches / CURR_SIZE)
			print(num_batches)
			print(num_curricula)
			num_batches_tot += num_batches
			num_curricula_tot += num_curricula
	print('Totals')
	print(num_batches_tot)
	print(num_curricula_tot)

def get_handles(my_hdf5, N):
	dt = h5py.special_dtype(vlen=str)
	valid = my_hdf5.require_dataset('valid', shape=(N,), dtype=np.bool)
	images = my_hdf5.require_dataset('images', shape=(N, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
	normals = my_hdf5.require_dataset('normals', shape=(N, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
	objects = my_hdf5.require_dataset('objects', shape=(N, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
	worldinfos = my_hdf5.require_dataset('worldinfo', shape=(N,), dtype=dt)
	agentactions = my_hdf5.require_dataset('actions', shape=(N,), dtype=dt)
	images2 = my_hdf5.require_dataset('images2', shape = (N, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype = np.uint8)
	normals2 = my_hdf5.require_dataset('normals2', shape = (N, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype = np.uint8)
	objects2 = my_hdf5.require_dataset('objects2', shape = (N, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype = np.uint8)
	return valid, images, normals, objects, worldinfos, agentactions, images2, normals2, objects2

def transfer_batch(load_hdf5, target_handles, target_bn, load_bn, valid_written_so_far):
	if target_bn < valid_written_so_far:
		print('skipping')	
		return
	valid, images, normals, objects, worldinfos, agentactions, images2, normals2, objects2 = target_handles
	desc_handle_pairs = [('valid', valid), ('images', images), ('normals', normals), ('objects', objects), ('worldinfo', worldinfos), ('actions', agentactions), ('images2', images2),
		('normals2', normals2), ('objects2', objects2)]
	for desc, handle in desc_handle_pairs:
		handle[(target_bn - valid_written_so_far) * BATCH_SIZE : (target_bn - valid_written_so_far + 1) * BATCH_SIZE] = load_hdf5[desc][load_bn * BATCH_SIZE : (load_bn + 1) * BATCH_SIZE]
	

def write_consolidated_dataset():
	f = h5py.File(big_hdf5_filename, mode = 'a')
	f_other = h5py.File(other_big_filename, mode = 'a')
	filenames_flat = [os.path.join(path, fn + '.hdf5') for (path, fns) in zip(paths, filenames) for fn in fns]
	target_bn = 0
	valid_num_batches_so_far = get_valid_num_batches(f)
	valid_num_batches_so_far -= 1
	print('in previous file, batches: ' + str(valid_num_batches_so_far))
	handles = get_handles(f_other, NUM_CURRICULA_TOT * CURR_SIZE * BATCH_SIZE - valid_num_batches_so_far * BATCH_SIZE)	
	for fn in filenames_flat:
		load_hdf5 = h5py.File(fn, mode = 'r')
		num_batches = get_valid_num_batches(load_hdf5)
		num_curricula = int(num_batches / CURR_SIZE)
		num_batches_full_curr = num_curricula * CURR_SIZE
		print(fn)
		print('num curricula: ' + str(num_curricula))
		print('num batches to transfer: ' + str(num_batches_full_curr))
		for load_bn in range(num_batches_full_curr):
			print('load bn: ' + str(load_bn))
			print('target bn: ' + str(target_bn))
			print('transferring')
			transfer_batch(load_hdf5, handles, target_bn, load_bn, valid_num_batches_so_far)
			print('flushing')
			f.flush()
			target_bn += 1
		load_hdf5.close()

write_consolidated_dataset()







	





