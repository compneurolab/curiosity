import make_new_tfrecord as mnt
import numpy as np
from PIL import Image
import os

TEST_SAVE_DIR = '/media/data3/new_dataset/test_images'

def write_images(batch_data, dir_name):
	full_dir_name = os.path.join(TEST_SAVE_DIR, dir_name)
	if os.path.isdir(full_dir_name):
		raise Exception('Directory already exists!')
	else:
		os.mkdir(full_dir_name)
        descs = ['images', 'objects', 'normals', 'images2', 'objects2', 'normals2']
	for desc in descs:
		print(desc)
		imagestuffs = batch_data[desc]
                for (i, img) in enumerate(imagestuffs):
                        print(i)
                        counter = str(i)
                        while len(counter) < 4:
                                counter = '0' + counter
                        fn = os.path.join(full_dir_name, desc + '_' + counter + '.png')
                        im = Image.fromarray(img)
                        im.save(fn)

batch_stuff = mnt.get_batch_data((0, 1234))

batch_stuff1 = mnt.get_batch_data((0, 0))
#write_images(batch_stuff, 'test1234')





