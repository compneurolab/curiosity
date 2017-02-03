'''
Just playing with quantization and one-hot encoding in tf, might want to refer to this later
'''

import tensorflow as tf
import numpy as np


# num_classes = 256
# my_tv = (num_classes - 1) * np.random.rand(128, 256, 256, 3)
# my_tv = tf.constant(my_tv, dtype = tf.uint8)
# my_tv = tf.one_hot(my_tv, depth = num_classes)
# my_pred = np.random.rand(128, 256, , 3, num_classes)
# my_pred = tf.constant(my_pred, dtype = tf.float32)

# # my_ten = tf.constant(my_arr, dtype = tf.float32)
# # quant = tf.cast(my_ten, tf.uint8)

# res = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(my_pred, my_tv))

my_ten = tf.constant([[1, 2, 3], [4, 5, 6]])
my_ten = my_ten - 1


# print(res)

with tf.Session() as sess:
	print(my_ten.eval())