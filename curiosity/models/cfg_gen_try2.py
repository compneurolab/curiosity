'''
Configuration generator, try 2
'''

import numpy as np
import cPickle
import os


SEED = 0


N_CHANNELS = 3 * 3
N_GAUSSIAN_CHANNELS = 11
N_TIMESTEPS_SEEN = 3
N_TIMESTEPS_PREDICT = 20
N_PARAMS_LIST = [25000 * (2**n) for n in range(1, 9)]
N_END = N_TIMESTEPS_PREDICT * 2
N_HIDDEN_LAYERS_LIST = [2, 2, 2, 2, 2, 4, 6, 8, 10]
N_ENCODING_LAYERS_LIST = [1, 2, 3, 3, 3, 3, 3, 4, 4, 4]
SIZE_STRIDE_CHOICES=  [(7, 2)] * 3 + [(5, 2)] * 3 + [(3, 2)] * 3 + [(3, 1)] * 1 + [(1, 1)]*1
N_FILTERS_LIST = [10 * n for n in range(1, 6)]
IMG_HEIGHT = 160
IMG_WIDTH = 375
WRITE_LOC = '/mnt/fs0/nhaber/projects/curiosity/curiosity/configs/cfg_gen_2.pkl'
STARTS_WRITE_LOC = '/mnt/fs0/nhaber/projects/temp/started_small2.pkl'

my_rng = np.random.RandomState(seed = SEED)


def get_next_to_start():
	if not os.path.exists(STARTS_WRITE_LOC):
		idx = 0
	else:
		with open(STARTS_WRITE_LOC) as stream:
			idx = cPickle.load(stream)
	with open(WRITE_LOC) as stream:
		retval = cPickle.load(stream)[idx]
	with open(STARTS_WRITE_LOC, 'w') as stream:
		cPickle.dump(idx + 1, stream)
	print('Starting config ' + str(idx))
	return 'gencfg_' + str(idx), retval

def get_config_list():
	with open(WRITE_LOC) as stream:
		return cPickle.load(stream)


def generate_encoding():
	cfg = {'size_1_before_concat_depth' : 1, 'size_2_before_concat_depth' : 0}
	size_first_encode = my_rng.choice([7, 5])
	size_max = size_first_encode
	num_filters_first_encode = my_rng.choice(N_FILTERS_LIST)
	cfg['size_1_before_concat'] = {
		1 : {'conv' : {'filter_size' : size_first_encode, 'stride' : 2, 'num_filters' : 24}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}}
	}
	cum_num_params = N_CHANNELS * num_filters_first_encode * size_first_encode**2 
	encode_depth = my_rng.choice(N_ENCODING_LAYERS_LIST)
	bypass_every_time = my_rng.choice([0, 1])
	channel_nums = [N_CHANNELS, num_filters_first_encode]
	cfg['encode_depth'] = encode_depth
	cfg['encode'] = {}
	for i in range(1, encode_depth + 1):
		sz_strd_idx = my_rng.choice([idx for idx, (sz, strd) in enumerate(SIZE_STRIDE_CHOICES) if sz <= size_max])
		k_size, stride = SIZE_STRIDE_CHOICES[sz_strd_idx] #dumb hack bc array for choice needs to be 1-dim
		size_max = k_size
		n_filters = my_rng.choice(N_FILTERS_LIST)
		channel_nums.append(n_filters)
		num_channels_in = channel_nums[-2]
		if bypass_every_time:
			bypass = 0
		else:
			bypass = my_rng.choice([0, 1, None])
			if bypass == 1:
				bypass = my_rng.choice(range(i + 1))
		if bypass is not None:
			num_channels_in +=  channel_nums[bypass]
		cum_num_params += num_channels_in * n_filters * k_size * k_size
		cfg['encode'][i] = {
			'conv' : {'filter_size' : k_size, 'stride' : stride, 'num_filters' : n_filters},
			'bypass' : bypass
		}
	return cum_num_params, cfg
		
def compute_final_encoding_shape(cfg, img_height, img_width):
	cum_stride = 4
	encode_depth = cfg['encode_depth']
	for i, enc_details in cfg['encode'].iteritems():
		cum_stride *= enc_details['conv']['stride']
	return (np.ceil(float(img_height) / float(cum_stride)), np.ceil(float(img_width) / float(cum_stride)), cfg['encode'][encode_depth]['conv']['num_filters'])


def generate_cfg():
	n_params = my_rng.choice(N_PARAMS_LIST)
	n_params_encode, cfg = generate_encoding()
	if cfg is None or n_params_encode > n_params:
		return n_params, n_params_encode, -1, (-1, -1),  None
	enc_dim = N_TIMESTEPS_SEEN * np.prod(compute_final_encoding_shape(cfg, IMG_HEIGHT, IMG_WIDTH))
	hidden_depth = my_rng.choice(N_HIDDEN_LAYERS_LIST)
	hidden_width = np.ceil( (- enc_dim - N_END + np.sqrt((enc_dim + N_END)**2 + 4 * hidden_depth * (n_params - n_params_encode)))
				/ (2 * hidden_depth))
	if hidden_width < N_END / 4:
		return n_params, n_params_encode, enc_dim, (hidden_depth, hidden_width), None
	cfg['hidden_depth'] = hidden_depth + 1
	cfg['hidden'] = {}
	for i in range(1, hidden_depth + 1):
		cfg['hidden'][i] = {'num_features' : hidden_width}
	cfg['hidden'][hidden_depth + 1] = {'num_features': N_END, 'activation' : 'identity'}
	return n_params, n_params_encode, enc_dim, (hidden_depth, hidden_width), cfg
	
ctr = 0

def generate_some_configs(how_many):
	ctr = 0
	configs = []
	while ctr < how_many:
		print('try')
		res = generate_cfg()
		if res[-1] is not None:
			configs.append(res[-1])
			ctr += 1
	with open(WRITE_LOC, 'w') as stream:
		cPickle.dump(configs, stream)
	return configs



def compute_histogram(cfgs, statistic_fn):
	histy = {}
	for cfg in cfgs:
		ans = statistic_fn(cfg)
		if ans in histy:
			histy[ans] += 1
		else:
			histy[ans] = 1
	return histy  

def get_configs(how_many):
	ctr = 0
	configs = []
	while ctr < how_many:
		print('try')
		res = generate_cfg()
		if res[-1] is not None:
			configs.append(res[-1])
			ctr += 1
	return configs

def print_histograms(how_many = 1000):
	get_encode_depth = lambda cfg : cfg['encode_depth']
	get_hidden_depth = lambda cfg : cfg['hidden_depth']
	get_filter_sizes = lambda cfg : str([cfg['size_1_before_concat'][1]['conv']['filter_size']] + [cfg['encode'][i]['conv']['filter_size'] 
						for i in range(1, cfg['encode_depth'] + 1)])
	get_prod_strides = lambda cfg : np.prod([cfg['encode'][i]['conv']['stride'] for i in range(1, cfg['encode_depth'] + 1)])
	configs = get_configs(how_many)
	encode_depth_histy = compute_histogram(configs, get_encode_depth)
	hidden_depth_histy = compute_histogram(configs, get_hidden_depth)
	filter_sizes_histy = compute_histogram(configs, get_filter_sizes)
	strides = compute_histogram(configs, get_prod_strides)
	print('encode depth')
	for i in sorted(encode_depth_histy.keys()):
		print(str(i) + ': ' + str(encode_depth_histy[i]))
	print('hidden depth histy')
	for k in sorted(hidden_depth_histy.keys()):
		print(str(k) + ': ' + str(hidden_depth_histy[k]))
	print('filter sizes histy')
	for k in sorted(filter_sizes_histy.keys()):
		print(str(k) + ': ' + str(filter_sizes_histy[k]))
	print('stride histy')
	for k in sorted(strides.keys()):
		print(str(k) + ': ' + str(strides[k]))
	 











if __name__ == '__main__':
	configs = generate_some_configs(100)


