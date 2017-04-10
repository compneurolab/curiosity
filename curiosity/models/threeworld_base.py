import numpy as np
import tensorflow as tf
import os
import cPickle

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
            print("Data statistics loaded.")
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

    def normalize(self, inputs):
        for source in self.methods:
            data = tf.cast(inputs[source], tf.float32)
            method = self.methods[source]

            #match dims for broadcasting
            while(len(self.stats[source]['mean'].shape) <
                    len(data.get_shape().as_list())):
                self.stats[source]['mean'] = \
                        np.expand_dims(self.stats[source]['mean'], 0)
                self.stats[source]['std'] = \
                        np.expand_dims(self.stats[source]['std'], 0)
                self.stats[source]['max'] = \
                        np.expand_dims(self.stats[source]['max'], 0)
                self.stats[source]['min'] = \
                        np.expand_dims(self.stats[source]['min'], 0)

            if method is 'standard':
                data = (data - self.stats[source]['mean']) / \
                          (self.stats[source]['std'] + self.epsilon)
            elif method is 'minmax':
                data = (data - self.stats[source]['min']) / \
                          (self.stats[source]['max'] - self.stats[source]['min'])
            else:
                raise ValueError('Unknown normalization method for %s' % source)

            inputs[source] = data
        return inputs

class ThreeWorldBaseModel:
    def __init__(self,
                 inputs,
                 gaussian=None,
                 normalization=None,
                 *args,
                 **kwargs):
        self.inputs = inputs
        self.gaussian = gaussian
        self.normalization = normalization

        # store reference to not normed inputs for not normed pixel positions
        inputs_not_normed = inputs

        if self.normalization is not None:
            self.inputs = self.normalization.normalize(self.inputs)

        if self.gaussian is not None:
            # get image shape
            for image_data in ['images', 'normals', 'objects', \
                            'images2', 'normals2', 'objects2']:
                if image_data in self.inputs:
                    image_shape = self.inputs[image_data].get_shape().as_list()
                    break
            # transform to 2d gaussian
            if 'poses' in self.gaussian:
                #object blobs
                gaussians = []
                centroids = tf.slice(inputs_not_normed['object_data'],\
                        [0, 0, 0, 8], [-1, -1, -1, 2])
                centroids = tf.unstack(centroids, axis=2)
                poses = tf.slice(self.inputs['object_data'],\
                        [0, 0, 0, 1], [-1, -1, -1, 4])
                poses = tf.unstack(poses, axis=3)
                for pose in poses:
                    pose = tf.unstack(pose, axis=2)
                    gaussians.append(\
                        self.create_gaussian_channel(image_shape, centroids, pose))
                self.inputs['object_data'] = tf.concat(gaussians, 4)

            if 'actions' in self.gaussian:
                # action blobs
                gaussians = []
                centroid = tf.slice(inputs_not_normed['actions'], [0, 0, 6], [-1, -1, 2])

                # TODO centroid scaled here because not done yet
                print('\033[91mWARNING: Actions are being scaled!\033[0m')
                centroid /= tf.reshape(tf.constant(
                    np.array([256.0 / 160.0, 600.0 / 375.0], dtype=np.float32)), 
                    [1,1,-1])

                forces = tf.slice(self.inputs['actions'], [0, 0, 0], [-1, -1, 6])
                forces = tf.unstack(forces, axis=2)
                for force in forces:
                    gaussians.append(\
                        self.create_gaussian_channel(image_shape, centroid, force))
                self.inputs['actions'] = tf.concat(gaussians, 4)

    def create_gaussian_channel(self, 
                                size, 
                                center=None, 
                                magnitude=None,
                                dtype = tf.float32,
                                fwhm = 10.0):
        '''
        size: kernel size
        fwhm: full-width-half-maximum (effective radius)
        center: kernel_center
        '''
        batch_size = size[0]
        sequence_len = size[1]
        width = size[2]
        height = size[3]
        channels = size[4]

        x = tf.range(0, width, 1, dtype=tf.float32)
        x = tf.tile(x, [batch_size * sequence_len])
        x = tf.reshape(x, [batch_size, sequence_len, width, 1, 1]) #column vector
        y = tf.range(0, height, 1, dtype=tf.float32)
        y = tf.tile(y, [batch_size * sequence_len])
        y = tf.reshape(y, [batch_size, sequence_len, 1, height, 1]) #row vector

        x0s = []
        y0s = []
        if center is None:
            x0s.append(tf.constant(width // 2))
            y0s.append(tf.constant(height // 2))
        elif isinstance(center, list):
            for c in center:
                x0s.append(tf.slice(c, [0, 0, 0], [-1, -1, 1]))
                y0s.append(tf.slice(c, [0, 0, 1], [-1, -1, 1]))
        else:
            x0s.append(tf.slice(center, [0, 0, 0], [-1, -1, 1]))
            y0s.append(tf.slice(center, [0, 0, 1], [-1, -1, 1]))

        mags = []
        if magnitude is None:
            mags.append(tf.ones([1, 1, 1]))
        elif isinstance(magnitude, list):
            mags = magnitude
        else:
            mags.append(magnitude)

        # create one dimensional channel with gaussians at [x0s, y0s]
        gauss = tf.zeros(size[0:4] + [1])
        for x0, y0, mag in zip(x0s, y0s, mags):
            x0 = tf.reshape(x0, [batch_size, sequence_len, 1, 1, 1])
            y0 = tf.reshape(y0, [batch_size, sequence_len, 1, 1, 1])
            mag = tf.reshape(mag, [batch_size, sequence_len, 1, 1, 1])
            gauss += tf.exp(-4.0*tf.log(2.0) * \
                    ((x-x0)**2.0 + (y-y0)**2.0) / fwhm**2.0) * mag
        return tf.cast(gauss, dtype)

def example_model(inputs,
                  batch_size=256,
                  gaussian=None,
                  stats_file=None,
                  normalization_method=None,
                  **kwargs):

    if normalization_method is not None:
        assert stats_file is not None, ('stats file has to be provided\
                to use normalization')
        normalization=Normalizer(stats_file, normalization_method)
    else:
        normalization=None

    actions = inputs['actions']
    net = ThreeWorldBaseModel(inputs, gaussian=gaussian, normalization=normalization);
    images = net.inputs['images']
    images = tf.cast(images, tf.float32)

    if gaussian is not None:
        images = tf.concat([images, net.inputs['object_data']], 4)
        images = tf.concat([images, net.inputs['actions']], 4)

    images = tf.slice(images, [0,0,0,0,0], [1,-1,-1,-1,-1])
    actions = tf.slice(actions, [0,0,0], [1,-1,-1])

    actions = tf.reshape(actions, [1, -1])
    flat_images = tf.reshape(images, [1, -1])
    W = tf.get_variable('W', [flat_images.get_shape().as_list()[-1], 
                              actions.get_shape().as_list()[-1]])
    out = tf.matmul(flat_images,W)

    outputs = {'images': out, 'actions': actions, 'concat_images': images}
    return [outputs, {}]

def dummy_loss(labels, logits, **kwargs):
    return tf.reduce_mean(logits['images'] - logits['actions'])
