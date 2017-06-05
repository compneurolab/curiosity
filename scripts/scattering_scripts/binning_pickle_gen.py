'''
A place to generate binning for our cross-entropy losses.

'''

import os
import cPickle

SAVE_DIR = '/mnt/fs0/nhaber/cross_ent_bins'

if not os.path.exists(SAVE_DIR):
	os.mkdir(SAVE_DIR)


EVEN_WEIGHTING_COARSE = [-.0228, .0228]
THREE_TO_ONE = [3., 1., 3.]
THREE_TO_ONE_SMALL = [1., .333, 1.]
TRY_TO_BALANCE_CLASSES = [.7 / 40 * float(n) for n in range(-10, 11)]


def make_even_weighting_bins(bin_cutoff_list, save_fn):
	weights = [1.] * (len(bin_cutoff_list) + 1)
	to_save = {'bins' : bin_cutoff_list, 'weights' : weights}
	with open(os.path.join(SAVE_DIR, save_fn), 'w') as stream:
		cPickle.dump(to_save, stream)
	return to_save

def make_bins(bin_cutoff_list, weights, save_fn):
	to_save = {'bins' : bin_cutoff_list, 'weights' : weights}
	with open(os.path.join(SAVE_DIR, save_fn), 'w') as stream:
		cPickle.dump(to_save, stream)
	return to_save


if __name__ == '__main__':
	saved3 = make_even_weighting_bins(TRY_TO_BALANCE_CLASSES, 'more_balanced_classes_try.pkl')
