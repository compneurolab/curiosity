import h5py
import numpy
import json
import os


paths = ['/media/data2/one_world_dataset', '/media/data/two_world_dataset']
filenames = [['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset4_resized', 'dataset5', 'dataset6', 'dataset8', 'dataset9_resized', 'dataset10', 'dataset11'], ['dataset1', 'dataset2', 'dataset3', 'dataset5', 'dataset7']]
big_hdf5_filename = '/media/data3/new_dataset/new_dataset.hdf5'

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
		num_curricula_in_path = 0
		n_batch_path = 0
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
			num_curricula_in_path += num_curricula
			n_batch_path += num_batches
		print('Total in path ' + path)
		print(num_curricula_in_path)
		print(n_batch_path)
        print('Totals')
        print(num_batches_tot)
        print(num_curricula_tot)


#total found in data2: 83 curr, 6053 batches
#total found in data1: 32 curr, 2250 batches

get_counts()
