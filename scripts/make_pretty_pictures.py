'''
Hacking to get some object and action images.
'''

import numpy as np
import make_new_tfrecord as mnt
from PIL import Image
import os
from sklearn.linear_model import LinearRegression

BATCH_SIZE = 256
SAVE_DIR = '/home/nhaber/temp/test_imgs'

def write_img(np_arr, filename_short):
	if not os.path.exists(SAVE_DIR):
		os.mkdir(SAVE_DIR)
	img = Image.fromarray(np_arr)
	print('got here')
	img.save(os.path.join(SAVE_DIR, filename_short + '.png'))

def get_some_positions_and_centroids(num_batches):
	batches_of_data = [mnt.get_batch_data((0, bn), with_non_object_images = False) for bn in range(num_batches)]
	positions = np.array([frame_obj_dat[0, 5:8] for batch_obj_dat in batches_of_data for frame_obj_dat in batch_obj_dat['object_data']])
	cms = np.array([frame_obj_dat[0, 8:] for batch_obj_dat in batches_of_data for frame_obj_dat in batch_obj_dat['object_data']])
	return positions, cms

def get_interesting_positions_and_centroids(begin_bn, end_bn):
	batches_of_data = [mnt.get_batch_data((0, bn), with_non_object_images = False) for bn in range(begin_bn, end_bn)]
	positions = []
	cms = []
	for batch_data in batches_of_data:
		for i in range(BATCH_SIZE):
			if batch_data['is_object_in_view'][i][0]:
				positions.append(batch_data['object_data'][i][0,5:8])
				cms.append(batch_data['object_data'][i][0,8:])
	return np.array(positions), np.array(cms)

def write_weird_images():
	num_batches = 1
	batches_of_data = [mnt.get_batch_data((0, bn), with_non_object_images = True) for bn in range(num_batches)]
	positions = []
	cms = []
	images = []
	objects = []
	objects1 = []
	ids = []
	for batch_data in batches_of_data:
		for i in range(BATCH_SIZE):
			if batch_data['is_object_in_view'][i][0]:
				positions.append(batch_data['object_data'][i][0,5:8])
				ids.append(batch_data['object_data'][i][0,0])
				cms.append(batch_data['object_data'][i][0,8:])
				images.append(batch_data['images'][i])
				oarray = batch_data['objects'][i]
				objects.append(oarray)
				oarray1 = 256**2 * oarray[:, :, 0] + 256 * oarray[:, :, 1] + oarray[:, :, 2]
				objects1.append(oarray1)
	for (ctr, my_pos) in enumerate(positions):
		if my_pos[2] < 0:
			print(ctr)
			print(my_pos)
			print cms[ctr]
			print(ids[ctr])
			my_img = images[ctr].astype(np.float32)
			my_img = my_img / 255.
			draw_exp(my_img, cms[ctr], sigma = 5.)
			my_img = make_viz_ready(my_img)
			write_img(my_img, 'weird' + str(ctr))
			oarray1 = objects1[ctr]
			highlighted_obj = (oarray1 == ids[ctr]).astype(np.float32)
			highlighted_obj = make_viz_ready(highlighted_obj)
			write_img(highlighted_obj, 'weird_obj' + str(ctr))
			oarray = objects[ctr]
			write_img(oarray, 'weird_all_objects' + str(ctr))
	return objects, objects1

def get_interesting_stuff_and_write_some():
	num_batches = 1
	batches_of_data = [mnt.get_batch_data((0, bn), with_non_object_images = True) for bn in range(num_batches)]
	positions = []
	cms = []
	images = []
	objects = []
	objects1 = []
	ids = []
	for batch_data in batches_of_data:
		for i in range(BATCH_SIZE):
			if batch_data['is_object_in_view'][i][0]:
				positions.append(batch_data['object_data'][i][0,5:8])
				ids.append(batch_data['object_data'][i][0,0])
				cms.append(batch_data['object_data'][i][0,8:])
				images.append(batch_data['images'][i])
				oarray = batch_data['objects'][i]
				objects.append(oarray)
				oarray1 = 256**2 * oarray[:, :, 0] + 256 * oarray[:, :, 1] + oarray[:, :, 2]
				objects1.append(oarray1)
	for (ctr, my_pos) in enumerate(positions):
		if ctr < 10:
			print(ctr)
			print(my_pos)
			print cms[ctr]
			print(ids[ctr])
			my_img = images[ctr].astype(np.float32)
			my_img = my_img / 255.
			draw_exp(my_img, cms[ctr], sigma = 5.)
			my_img = make_viz_ready(my_img)
			write_img(my_img, 'im' + str(ctr))
			# oarray1 = objects1[ctr]
			# highlighted_obj = (oarray1 == ids[ctr]).astype(np.float32)
			# highlighted_obj = make_viz_ready(highlighted_obj)
			# write_img(highlighted_obj, 'weird_obj' + str(ctr))
			# oarray = objects[ctr]
			# write_img(oarray, 'weird_all_objects' + str(ctr))
	return positions, cms


def play_with_pos_and_cms(positions, cms):
	cms_centered = cms - np.array([[160./2., 375. / 2.] for _ in range(len(cms))])
	positions_divided = np.array([[pos[0] / (pos[2]), pos[1] / (pos[2])] for pos in positions])
	for (ctr, (pos, cm)) in enumerate(zip(positions_divided, cms_centered)):
		print(ctr)
		print(cm[1] / pos[0])
		print(cm[0] / pos[1])

def solve_for_f_and_h(positions, cms):
	cms_centered = cms - np.array([[160./2., 375. / 2.] for _ in range(len(cms))])
	ratio0 = np.array([[cm[0] / pos[1]] for (cm, pos) in zip(cms_centered, positions)])
	y0 = np.array([-cm[0] * pos[2] / pos[1] for (cm, pos) in zip(cms_centered, positions)])
	ratio1 = np.array([[cm[1] / pos[0]] for (cm, pos) in zip(cms_centered, positions)])
	y1 = np.array([-cm[1] * pos[2] / pos[0] for (cm, pos) in zip(cms_centered, positions)])
	my_lr = LinearRegression()
	my_lr.fit(ratio0, y0)
	print my_lr.coef_
	print my_lr.intercept_
	print my_lr.residues_
	my_lr.fit(ratio1, y1)
	print my_lr.coef_
	print my_lr.intercept_
	print my_lr.residues_

def solve_try_2(positions, cms):
	cms_centered = cms - np.array([[160./2., 375. / 2.] for _ in range(len(cms))])
	X0 = np.array([[p[1], -c[0]] for (c, p) in zip(cms_centered, positions)])
	Y0 = np.array([c[0] * p[2] for (c, p) in zip(cms_centered, positions)])
	X1 = np.array([[p[0], -c[1]] for (c, p) in zip(cms_centered, positions)])
	Y1 = np.array([c[1] * p[2] for (c, p) in zip(cms_centered, positions)])
	my_lr = LinearRegression(fit_intercept = False)
	my_lr.fit(X0, Y0)
	print my_lr.coef_
	print my_lr.intercept_
	print my_lr.residues_
	my_lr.fit(X1, Y1)
	print my_lr.coef_
	print my_lr.intercept_
	print my_lr.residues_



def test_nearest_is_ok(num_batches):
	batches_of_data = [mnt.get_batch_data((0, bn), with_non_object_images = True) for bn in range(num_batches)]
	positions = []
	cms = []
	images = []
	objects = []
	objects1 = []
	ids = []
	for batch_data in batches_of_data:
		for i in range(BATCH_SIZE):
			if batch_data['is_object_in_view'][i][0]:
				positions.append(batch_data['object_data'][i][0,5:8])
				ids.append(batch_data['object_data'][i][0,0])
				cms.append(batch_data['object_data'][i][0,8:])
				images.append(batch_data['images'][i])
				oarray = batch_data['objects'][i]
				objects.append(oarray)
				oarray1 = 256**2 * oarray[:, :, 0] + 256 * oarray[:, :, 1] + oarray[:, :, 2]
				objects1.append(oarray1)
	for (ctr, my_pos) in enumerate(positions):
		print(ctr)
		print(my_pos)
		print cms[ctr]
		print(ids[ctr])
		my_img = images[ctr].astype(np.float32)
		my_img = my_img / 255.
		draw_exp(my_img, cms[ctr], sigma = 5.)
		my_img = make_viz_ready(my_img)
		write_img(my_img, 'new_img' + str(ctr))
		oarray1 = objects1[ctr]
		highlighted_obj = (oarray1 == ids[ctr]).astype(np.float32)
		highlighted_obj = make_viz_ready(highlighted_obj)
		write_img(highlighted_obj, 'new_obj' + str(ctr))
		oarray = objects[ctr]
		write_img(oarray, 'new_all_objects' + str(ctr))
	return objects, objects1


def draw_exp(np_arr_img, pos, sigma = 1.):
	for i in range(np_arr_img.shape[0]):
			for j in range(np_arr_img.shape[1]):
				for k in range(np_arr_img.shape[2]):
					np_arr_img[i, j, k] = min(1., np.exp(- np.linalg.norm(pos - np.array([i, j], dtype = np.float32))**2 / sigma) + np_arr_img[i, j, k])
	return np_arr_img

def make_viz_ready(np_arr_img):
	scaled = 255. * np_arr_img
	return scaled.astype(np.uint8)

def save_a_centroid():
	positions, cms = get_some_positions_and_centroids(1)
	original_img = np.zeros((160, 375), dtype = np.float32)
	new_img = make_viz_ready(draw_exp(original_img, cms[22], sigma = 10.))
	write_img(new_img, 'test22')


pos, cms = get_interesting_positions_and_centroids(500, 1000)
solve_try_2(pos, cms)
# objects, objects1 = write_weird_images()
