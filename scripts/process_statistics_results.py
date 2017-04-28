import numpy as np
import cPickle
from PIL import Image
import os

SAVE_DIR = '/media/data2/two_world_dataset/statistics'
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, 'images')

LOAD_FN = '/mnt/fs0/datasets/two_world_dataset/statistics/stats_updated.pkl'
SAVE_FN = '/mnt/fs0/datasets/two_world_dataset/statistics/stats_again.pkl'

def load_results(results_num):
	with open(os.path.join(SAVE_DIR, 'partition_' + str(results_num) + '.p')) as stream:
		return cPickle.load(stream)

def write_image(results_num):
	if not os.path.exists(IMAGE_SAVE_DIR):
		os.mkdir(IMAGE_SAVE_DIR)
	(statistics, num_seen) = load_results(results_num)
	im = Image.fromarray(statistics['images'][0].astype(np.uint8))
	im.save(os.path.join(IMAGE_SAVE_DIR, 'mean_' + str(results_num) + '.png'))

def shoot_wow_bad_forgetting():
	with open(LOAD_FN) as stream:
		stats_old = cPickle.load(stream)
	stats_new = {}
	for k in stats_old:
		stats_new[k] = stats_old[k]
		std = stats_old[k]['std']
		mean = stats_old[k]['mean']
		var_new = std**2 - mean**2
		stats_new[k]['std'] = np.sqrt(var_new)
	with open(SAVE_FN, 'w') as stream:
		cPickle.dump(stats_new, stream)
	

