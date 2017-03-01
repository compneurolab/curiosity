'''
Some tools for possible ways to mix in action data with spatial data.
'''

import numpy as np
import tensorflow as tf

from curiosity.models.model_building_blocks import ConvNetwithBypasses


		    # kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
		    #                          shape=[ksize1, ksize2, in_shape, out_shape],
		    #                          dtype=tf.float32,
		    #                          regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
		    #                          name='weights')


            # kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
            #                          shape=[in_shape, out_shape],
            #                          dtype=tf.float32,
            #                          name='weights')
            # biases = tf.get_variable(initializer=tf.constant_initializer(bias),
            #                          shape=[out_shape],
            #                          dtype=tf.float32,
            #                          name='bias')

