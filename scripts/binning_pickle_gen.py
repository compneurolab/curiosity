'''
A place to generate binning for our cross-entropy losses.

'''

import os
import cPickle

SAVE_DIR = '/mnt/fs0/nhaber/cross_ent_bins'

if not os.path.exists(SAVE_DIR):
	os.mkdir(SAVE_DIR)


EVEN_WEIGHTING_COARSE = [-.0228, .228]

def make_even_weighting_bins(bin_cutoff_list, save_fn):
	weights = [1.] * (len(bin_cutoff_list) + 1)
	to_save = {'bins' : bin_cutoff_list, 'weights' : weights}
	with open(os.path.join(SAVE_DIR, save_fn), 'w') as stream:
		cPickle.dump(to_save, stream)
	return to_save




if __name__ == '__main__':
	saved = make_even_weighting_bins(EVEN_WEIGHTING_COARSE, 'even_weighting_coarse.pkl')
