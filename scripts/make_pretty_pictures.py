'''
Hacking to get some object and action images.
'''

import numpy as np
import make_new_tfrecord as mnt

def get_some_positions_and_centroids(num_batches):
	batches_of_data = [mnt.get_batch_data((0, bn), with_non_object_images = False) for bn in range(num_batches)]
	positions = np.array([frame_obj_dat[0, 5:8] for batch_obj_dat in batches_of_data for frame_obj_dat in batch_obj_dat['object_data']])
	cms = np.array([frame_obj_dat[0, 8:] for batch_obj_dat in batches_of_data for frame_obj_dat in batch_obj_dat['object_data']])
	return positions, cms

positions, cms = get_some_positions_and_centroids(1)