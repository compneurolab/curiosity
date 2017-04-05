import numpy as np
import cPickle
from PIL import Image
import os

SAVE_DIR = '/media/data2/two_world_dataset/statistics'
IMAGE_SAVE_DIR = os.path.join(SAVE_DIR, 'images')

def load_results(results_num):
	with open(os.path.join(SAVE_DIR, 'partition_' + str(results_num) + '.p')) as stream:
		return cPickle.load(stream)

def write_image(results_num):
	if not os.path.exists(IMAGE_SAVE_DIR):
		os.mkdir(IMAGE_SAVE_DIR)
	(statistics, num_seen) = load_results(results_num)
	im = Image.fromarray(statistics['images'][0].astype(np.uint8))
	im.save(os.path.join(IMAGE_SAVE_DIR, 'mean_' + str(results_num) + '.png'))

(res0, numseen0) = load_results(0)
(res4, numseen4) = load_results(4)

print(numseen0)
print(numseen4)
write_image(0)
write_image(4)
	

