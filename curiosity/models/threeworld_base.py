import numpy as np
import tensorflow as tf
import os

class Normalizer:
    # epsilon used during normalization to avoid division by zero
    epsilon = np.array(1e-4).astype(np.float32)

    # dict mapping 'source' to 'statistics'
    stats = {}
    # dict mapping 'source' to 'normalization method'
    methods = {}

    def __init__(self,
                 stats_file,
                 normalization_methods):
        # open the stats file
        assert os.path.isfile(stats_file), ('stats file not a file: %s' % stats_file)
        assert stats_file.endswith('.pkl'), ('stats file must be a .pkl file')
        with open(stats_file) as f:
            self.stats = cPickle.load(f)
        assert isinstance(self.stats, dict), ('stats must be a dictionary')

        # check if normalization methods available in stats file and convert to float32
        assert isinstance(normalization_methods, dict),\
            ('normalization methods must be a dict that maps \'source\'\
            to \'normalization method\'')
        self.methods = normalization_methods
        for source in self.methods:
            assert source in self.stats,\
                    ('%s not present in stats file', source)

            for k in ['mean', 'std', 'min', 'max']:
                assert k in self.stats[source], ('%s not present in %s' % (k, source))

            for k in self.stats[source]:
                self.stats[source][k] = \
                        self.stats[source][k].astype(np.float32)

                # if both min=0 and max=0 set max=1 to avoid dividing by 0
                # during minmax norm
                if k is 'max':
                    idx = np.where(self.stats[source]['min'] == 0)
                    _max = self.stats[source]['max']
                    _max[(idx[0][_max[idx] == 0], idx[1][_max[idx] == 0])] = 1.0

    def normalize(self, data, source):
        method = self.methods(source)
        if method is 'standard':
            data = (data - self.stats[source]['mean']) / \
                      (self.stats[source]['std'] + self.epsilon)
        elif method is 'minmax':
            data = (data - self.stats[source]['min']) / \
                      (self.stats[source]['max'] - self.stats[source]['min'])
        else:
            raise ValueError('Unknown normalization method for %s' % source)
        return data

class ThreeWorldBaseModel:
    def __init__(self,
                 gaussian=None,
                 normalization=None,
                 *args,
                 **kwargs):
        self.gaussian = gaussian
        self.normalization = normalization
        
        for inp in inputs:
            if inp in self.normalization.method:
                data = self.normalization.normalize(data, inp)

    def create_gaussian_kernel(self, size, center=None, fwhm = 10.0):
        '''
        size: kernel size
        fwhm: full-width-half-maximum (effective radius)
        center: kernel_center
        '''
        batch_size = size[0]
        x = tf.range(0, size[1], 1, dtype=tf.float32)
        x = tf.tile(x, [batch_size])
        x = tf.reshape(x, [batch_size, 1, size[1]]) #column vector
        y = tf.range(0, size[2], 1, dtype=tf.float32)
        y = tf.tile(y, [batch_size])
        y = tf.reshape(y, [batch_size, size[2], 1]) #row vector

        if center is None:
            x0 = tf.constant(size[1] // 2)
            y0 = tf.constant(size[2] // 2)
        else:
            x0 = tf.slice(center, [0, 0], [-1, 1])
            y0 = tf.slice(center, [0, 1], [-1, 1])

        x0 = tf.reshape(x0, [batch_size, 1, 1])
        y0 = tf.reshape(y0, [batch_size, 1, 1])
        gauss = tf.exp(-4.0*tf.log(2.0) * ((x-x0)**2.0 + (y-y0)**2.0) / fwhm**2.0)
        y0 = tf.reshape(y0, [batch_size, 1, 1])
        gauss = tf.exp(-4.0*tf.log(2.0) * ((x-x0)**2.0 + (y-y0)**2.0) / fwhm**2.0)
        size.append(1) # append channel dimension
        gauss = tf.reshape(gauss*255, size) # scale to whole uint8 range
        return tf.cast(gauss, tf.uint8)

    def convert_to_2d_gaussian(self, data, source):
        raise NotImplementedError()

def example_model(inputs,
                  batch_size=256,
                  **kwargs):
    images = inputs['images']
    actions = inputs['actions']

    images = tf.slice(images, [0,0,0,0,0], [1,-1,-1,-1,-1])
    actions = tf.slice(actions, [0,0,0], [1,-1,-1])

    actions = tf.reshape(actions, [1, -1])
    images = tf.reshape(images, [1, -1])
    images = tf.cast(images, tf.float32)
    W = tf.get_variable('W', [images.get_shape().as_list()[-1], 
                              actions.get_shape().as_list()[-1]])
    images = tf.matmul(images,W)

    outputs = {'images': images, 'actions': actions}
    return [outputs, {}]

def dummy_loss(labels, logits, **kwargs):
    return tf.reduce_mean(logits['images'] - logits['actions'])
