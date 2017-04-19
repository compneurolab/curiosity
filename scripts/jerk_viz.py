'''
jerk viz
'''

import make_new_tfrecord as mnt

BATCH_SIZE = 256

def derivative(my_dict, times):
	valid_times = times[1:-1]
	return dict((t, (my_dict[t + 1] - my_dict[t - 1]) / 2.) for t in valid_times), valid_times

def display_jerk_batch(bn):
	batch_data = mnt.get_batch_data((0, bn), with_non_object_images = False)
	positions = dict((t, dat[0][5:8]) for t, dat in enumerate(batch_data['object_data']))
	pos_times = range(BATCH_SIZE)
	velocity, vel_times = derivative(positions, pos_times)
	accel, accel_times = derivative(velocity, vel_times)
	jerk, jerk_times = derivative(accel, accel_times)
	print jerk.keys()
	print jerk.values()
	for t in range(BATCH_SIZE):
		print(t)
		print 'object there: ' + str(batch_data['is_object_there'][t][0])
		print('not teleporting: ' + str(batch_data['is_not_teleporting'][t][0]))
		if t in jerk:
			print jerk[t]
		else:
			print('no jerk')




if __name__ == '__main__':
	display_jerk_batch(0)
