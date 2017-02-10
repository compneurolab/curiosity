'''
Running a model that's meant to do timestep 1 evolution in the hidden layers.
'''


cfg_alexy = {
	'encode_depth' : 5,
	'decode_depth' : 5,
	'hidden_depth' : 3,
	'encode' : {
		1 : {'conv' : {'filter_size' : 11, 'stride' : 4, 'num_filters' : 96}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 1, 'num_filters' : 256}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 384}},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 384}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 256}}, # size 256 image, this leads to 16 * 16 * 256 = 65,536 neurons. Sad!
	}



	'decode' : {
		0 : {'size' : 16, 'num_filters' : 256},
		1 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 384},
		2 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 384},
		3 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 256},
		4 : {'filter_size' : 5, 'size' : 32, 'num_filters' : 96},
		5 : {'filter_size' : 11, 'size' : 256, 'num_filters' : 3}
	}

	'hidden' : {
		1 : {'num_features' : 256},
		2 : {'num_features' : 256},
		3 : {'num_features' : 65,536}
	}

}

cfg_alexy_small = {
	'encode_depth' : 5,
	'decode_depth' : 5,
	'hidden_depth' : 4,
	'encode' : {
		1 : {'conv' : {'filter_size' : 7, 'stride' : 2, 'num_filters' : 96}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		2 : {'conv' : {'filter_size' : 5, 'stride' : 2, 'num_filters' : 96}, 'pool' : {'size' : 3, 'stride' : 2, 'type' : 'max'}},
		3 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 96}},
		4 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 64}},
		5 : {'conv' : {'filter_size' : 3, 'stride' : 1, 'num_filters' : 32}}, # Size 16 image, giving 16 * 16 * 32 = 8192 neurons. Let's try it!
	}

	'decode' : {
		0 : {'size' : 16, 'num_filters' : 32},
		1 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 64},
		2 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 96},
		3 : {'filter_size' : 3, 'size' : 16, 'num_filters' : 96},
		4 : {'filter_size' : 5, 'size' : 64, 'num_filters' : 96},
		5 : {'filter_size' : 11, 'size' : 256, 'num_filters' : 3}
	}

	'hidden' : {
		1 : {'num_features' : 256},
		2 : {'num_features' : 256},
		3 : {'num_features' : 256},
		4 : {'num_features' : 8192}
	}

}