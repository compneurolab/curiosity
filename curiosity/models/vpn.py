import numpy as np
import tensorflow as tf
import copy
from tqdm import trange
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.framework.ops import Tensor

from curiosity.models.threeworld_base import ThreeWorldBaseModel

class BasicConvLSTMCell(object):
    """
    Basic Convolutional LSTM recurrent network cell. The
    """
    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, 
            state_is_tuple=False, activation=tf.nn.tanh, dtype=tf.float32):
            """Initialize the basic Conv LSTM cell.
            Args:
            shape: list(int d0, ... int dk) dimensions of cell
            filter_size: tuple(int height, int width) of filter
            num_features: (int num_features)  of cell ("depth")
            forget_bias: (float bias) added to forget gates
            state_is_tuple: (bool state_is_tuple) 
                If True, accepted and returned states are 2-tuples of
                the `c_state` and `m_state`.
                If False, they are concatenated along the column axis.
            activation: (func activation_function) of inner states.
            """
            self.shape = shape 
            self.filter_size = filter_size
            self.num_features = num_features
            self.dtype = dtype
            self._forget_bias = forget_bias
            self._state_is_tuple = state_is_tuple
            self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Parameters of gates are concatenated into one multiply for efficiency
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
            concat = self._conv_linear([inputs, h], \
                    self.filter_size, self.num_features * 4, True)
            
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                    self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=3, values=[new_c, new_h])
            return new_h, new_state

    def _conv_linear(self, args, filter_size, num_features, 
            bias, bias_start=0.0, scope=None):
        """convolution:
        Args:
            args: 4D Tensor or list of 4D, batch x n, Tensors.
            filter_size: tuple(int height, int width) of filter
            num_features: (int num_features) number of features.
            bias_start: starting value to initialize the bias; 0 by default.
            scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
            A 4D Tensor with shape [batch h w num_features]
        Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.
        """
        
        # Calculate the total size of arguments on dimension 1.
        total_arg_size_depth = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 4:
                raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
            if not shape[3]:
                raise ValueError("Linear expects shape[4]\
                        of arguments: %s" % str(shapes))
            else:
                total_arg_size_depth += shape[3]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope or "Conv") as conv_scope:
            #TODO Initialize variables before __call__
            try:
                matrix = tf.get_variable("Matrix", [filter_size[0], filter_size[1], \
                        total_arg_size_depth, num_features], dtype=dtype)
                bias_term = tf.get_variable("Bias", [num_features], dtype=dtype,
                        initializer=tf.constant_initializer(bias_start, dtype=dtype))
            except ValueError:
                conv_scope.reuse_variables()
                matrix = tf.get_variable("Matrix", [filter_size[0], filter_size[1], \
                        total_arg_size_depth, num_features], dtype=dtype)
                bias_term = tf.get_variable("Bias", [num_features], dtype=dtype,
                        initializer=tf.constant_initializer(bias_start, dtype=dtype))

            if len(args) == 1:
                res = tf.nn.conv2d(args[0], matrix, \
                        strides=[1, 1, 1, 1], padding='SAME')
            else:
                res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, \
                        strides=[1, 1, 1, 1], padding='SAME')
            if not bias:
                return res
        return res + bias_term
    
    def zero_state(self):
        """Return zero-filled state tensor(s).
        Returns:
            tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
            filled with zeros
        """
        # last dimension is replaced by 2 * num_features = (c, h)
        if self._state_is_tuple:
            state = LSTMStateTuple(
                    tf.zeros(self.shape[:-1] + [self.num_features]),
                    tf.zeros(self.shape[:-1] + [self.num_features]))
        else:
            state = tf.zeros(self.shape[:-1] + [self.num_features * 2], dtype=self.dtype)
        return state

class VPN(ThreeWorldBaseModel):
    lstm_initialized = False

    def new_variable(self, name, shape, dtype=tf.float32, seed=0):
        return tf.get_variable(
                name,
                shape=shape,
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                regularizer=None,
                )

    def mu(self, inputs, kernel_h=3, kernel_w=3, out_channels=128, \
            mask=None, conditioning=[], use_conditioning=True, scope='mu'):
        in_channels = inputs.get_shape().as_list()[-1]
        with tf.variable_scope(scope) as mu_scope:
            # resolution preserving stride
            strides = [1,1,1,1]          
            # parallelizing the convolution of W1-4
            W = self.new_variable('W', \
                    [kernel_h, kernel_w, in_channels, out_channels*4])
            # masking
            if isinstance(mask, Tensor):
                assert mask.get_shape().as_list() == W.get_shape().as_list(), \
                        ('mask must be of shape', W.get_shape().as_list())
                W *= mask
            elif isinstance(mask, np.ndarray):
                assert list(mask.shape) == W.get_shape().as_list(), \
                        ('mask must be of shape', W.get_shape().as_list())
                W *= mask
            elif mask in ['a', 'b']:
                # Use only for testing as this is not memory efficient
                # rather create your own mask and reuse it for multiple mus
                mask = np.ones(W.get_shape().as_list())
                mask[kernel_h // 2, kernel_w // 2 + 1:, :, :] = 0.0
                mask[kernel_h // 2 + 1:, :, :, :] = 0.0
                if mask is 'a':
                    mask[kernel_h // 2, kernel_w // 2, :, :] = 0.0
                W *= mask
            elif mask is not None:
                raise ValueError('mask has to be \'a\', \'b\', Tensor, or None')
            # (masked) convolution
            g = tf.nn.conv2d(inputs, W, strides=strides, padding='SAME')
            # conditioning
            for i, cond in enumerate(conditioning):
                cond_channels = cond.get_shape().as_list()[-1]
                V = self.new_variable('V' + str(i), \
                    [kernel_h, kernel_w, cond_channels, out_channels*4])
                if use_conditioning:
                    g += tf.nn.conv2d(cond, V, strides=strides, padding='SAME')
            # multiplicative unit        
            g1, g2, g3, g4 = tf.split(g, num_or_size_splits=4, axis=3)
            g1 = tf.sigmoid(g1)
            g2 = tf.sigmoid(g2)
            g3 = tf.sigmoid(g3)
            u = tf.tanh(g4)
            return g1 * tf.tanh(g2 * inputs + g3 * u)

    def rmb(self, inputs, kernel_h=1, kernel_w=1, mu_channels=128, \
            mask=None, conditioning=[], use_conditioning=True, scope='rmb'):
        in_channels = inputs.get_shape().as_list()[-1]
        out_channels = in_channels
        with tf.variable_scope(scope) as rmb_scope:
            W1 = self.new_variable('W1', 
                    [kernel_h, kernel_w, in_channels, mu_channels])
            W4 = self.new_variable('W4', 
                    [kernel_h, kernel_w, mu_channels, out_channels])

            strides=[1,1,1,1]
            h1 = tf.nn.conv2d(inputs, W1, strides=strides, padding='SAME')
            h2 = self.mu(h1, conditioning=conditioning, mask=mask, \
                    out_channels=mu_channels, \
                    use_conditioning=use_conditioning, scope='mu1') #W2 internal
            h3 = self.mu(h2, conditioning=conditioning, mask=mask, \
                    out_channels=mu_channels, \
                    use_conditioning=use_conditioning, scope='mu2') #W3 internal
            h4 = tf.nn.conv2d(h3, W4, strides=strides, padding='SAME')

            return inputs + h4

    def encoder(self, inputs, conditioning, num_rmb=8, scope='encoder'):
        print('Encoder: %d RMB layers' % num_rmb)
        inputs = self.ph_enc_inp = tf.placeholder_with_default(inputs, 
                inputs.get_shape().as_list(), 'enc_inp')
        #conditioning = self.ph_enc_cond = tf.placeholder_with_default(conditioning, 
        #        inputs.get_shape().as_list()[0:-1] + [256], 'enc_cond')
        with tf.variable_scope(scope) as encode_scope:
            # Residual Multiplicative Blocks
            inputs = tf.unstack(inputs, axis=1)
            if isinstance(conditioning, Tensor):
                conditioning = tf.unstack(conditioning, axis=1)
                assert len(conditioning) == len(inputs)
            for i, inp in enumerate(inputs):
                for r in range(num_rmb):
                    if conditioning:
                        inp = self.rmb(inp, scope = 'rmb' + str(r), \
                                conditioning = [conditioning[i]])
                    else:
                        inp = self.rmb(inp, scope = 'rmb' + str(r), \
                                conditioning = [])
                # share variables across frames
                encode_scope.reuse_variables()
        return tf.stack(inputs, axis=1)

    def lstm(self, inputs):
        #TODO make work nicely with dynamic rnn lengths
            #outputs, state = tf.nn.dynamic_rnn(
            #        self.conv_lstm_cell,
            #        inputs,
            #        sequence_length=None,
            #        dtype=tf.float32,
            #        initial_state=None,
            #        scope='ConvLSTM')
        inputs = self.ph_lstm_inp = tf.placeholder_with_default(inputs, 
                inputs.get_shape().as_list(), 'lstm_inp')
        inputs = tf.unstack(inputs, axis=1)
        assert len(inputs) > 0, ('input_len = ' + len(inputs) + '< 0')
        # Convolutional LSTM over time
        if not self.lstm_initialized:
            self.conv_lstm_cell = BasicConvLSTMCell(inputs[0].get_shape().as_list(),
                    [1, 1], 256, state_is_tuple=True)
            self.lstm_state = self.conv_lstm_cell.zero_state()
            self.lstm_outputs = []
            self.lstm_initialized = True
        for inp in inputs:
            output, self.lstm_state = self.conv_lstm_cell(inp, self.lstm_state)
            self.lstm_outputs.append(tf.expand_dims(output, axis = 1))
        return tf.concat(self.lstm_outputs, axis = 1)

    def decoder(self, inputs, conditioning, condition_first_image=False, \
            out_channels=768, num_rmb=12, scope='decoder', disable_print=False):
        if not disable_print:
            print('Decoder: %d RMB layers' % num_rmb)
        inputs = self.ph_dec_inp = tf.placeholder_with_default(inputs, 
                inputs.get_shape().as_list(), 'dec_inp')
        conditioning = self.ph_dec_cond = tf.placeholder_with_default(conditioning, 
                inputs.get_shape().as_list()[0:-1] + [256], 'dec_cond')
        with tf.variable_scope(scope) as decode_scope:
            # Residual Multiplicative Blocks
            inputs = tf.unstack(inputs, axis=1)
            if isinstance(conditioning, Tensor):
                conditioning = tf.unstack(conditioning, axis=1)
            # construct masking sequence
            mu_kernel_h = 3
            mu_kernel_w = 3
            mu_in_channels = 128
            mu_out_channels = 128
            maskB = np.ones([mu_kernel_h, mu_kernel_w, 
                mu_in_channels, mu_out_channels*4])
            maskB[mu_kernel_h // 2, mu_kernel_w // 2 + 1:, :, :] = 0.0
            maskB[mu_kernel_h // 2 + 1:, :, :, :] = 0.0
            maskA = copy.deepcopy(maskB)
            maskA[mu_kernel_h // 2, mu_kernel_w // 2, :, :] = 0.0
            maskA = tf.constant(maskA, dtype=tf.float32)
            maskB = tf.constant(maskB, dtype=tf.float32)
            # first rmb has mask 'a', subsequent rmbs have mask 'b'
            masks = [maskA] + [maskB] * (num_rmb - 1)
            outputs = []
            for i, inp in enumerate(inputs):
                W_in = self.new_variable('W_in', [1, 1, 3, 256])
                inp = tf.nn.conv2d(inp, W_in, strides=[1, 1, 1, 1], padding='SAME')
                for r, mask in enumerate(masks):
                    # first frame has no previous time steps to condition on
                    if i == 0:
                        inp = self.rmb(inp, scope = 'rmb' + str(r), \
                                conditioning = [conditioning[i]],
                                use_conditioning = condition_first_image,
                                mask=mask)
                    # subesequent frames condition on previous time steps
                    else:
                        inp = self.rmb(inp, scope = 'rmb' + str(r), \
                                conditioning = [conditioning[i-1]],
                                mask=mask)
                W_out = self.new_variable('W_out', [1, 1, 256, out_channels])
                inp = tf.nn.conv2d(inp, W_out, strides=[1, 1, 1, 1], padding='SAME')
                outputs.append(inp)
                # share variable across frames
                decode_scope.reuse_variables()
            outputs = tf.stack(outputs, axis=1)
        return outputs

    def reshape_rgb(self, inputs, out_channels):
        # reshape to [batch_size, time_step, height, width, n_channels, intensities]
        rgb_shape = inputs.get_shape().as_list()
        rgb_shape[-1] = out_channels / 256
        rgb_shape.append(256)
        return tf.reshape(inputs, rgb_shape)


    def get_intensities(self, inputs):
        inputs = tf.nn.softmax(inputs)
        inputs = tf.unstack(inputs)
        for i, inp in enumerate(inputs):
            shape = inp.get_shape().as_list()
            inputs[i] = tf.cast(tf.argmax(inp, axis=tf.rank(inp)-1), dtype=tf.uint8)
            inputs[i].set_shape(shape[0:-1])
            inputs[i] = tf.image.convert_image_dtype(inputs[i], dtype=tf.float32)
        inputs = tf.stack(inputs)
        return inputs

    def train(self, encoder_depth=8, decoder_depth=12, out_channels=768):
        images = self.inputs['images']
        # convert images to float32 between [0,1) if not normalized
        if self.normalization is None:
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        # encode
        encoded_inputs = self.encoder(images, conditioning=[], num_rmb=encoder_depth)
        # run lstm
        lstm_out = self.lstm(encoded_inputs)
        # decode
        rgb = self.decoder(images, conditioning=lstm_out, num_rmb=decoder_depth, 
                out_channels=out_channels)
        # reshape to [batch_size, time_step, height, width, n_channels, intensities]
        rgb = self.reshape_rgb(rgb, out_channels)
        return [{'rgb': rgb}, 
                {'encoder_depth': encoder_depth, 'decoder_depth': decoder_depth}]

    def test(self, encoder_depth=8, decoder_depth=12,
            out_channels=768, num_context=2):
        # convert images to float32 between [0,1) if not normalized
        if self.normalization is None:
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        # encode
        encoded_inputs = self.encoder(images, conditioning=[], num_rmb=encoder_depth)
        # run lstm
        lstm_out = self.lstm(encoded_inputs)
        # decode
        rgb = self.decoder(images, conditioning=lstm_out, num_rmb=decoder_depth,
                out_channels=out_channels)
        # reshape to [batch_size, time_step, height, width, n_channels, intensities]
        rgb = self.reshape_rgb(rgb, out_channels)
        # get intensities
        predicted_images = self.get_intensities(rgb)
        return [{'decode': predicted_images, 'run_lstm': lstm_out, 
            'encode': encoded_inputs, 'rgb': rgb},
            {'encoder_depth': encoder_depth, 'decoder_depth': decoder_depth}]

    def test_unroll_all_on_gpu(self, encoder_depth=8, decoder_depth=12,
            out_channels=768, num_context=2):
        tf.get_variable_scope().reuse_variables()
        images = self.inputs['images']
        # convert images to float32 between [0,1) if not normalized
        if self.normalization is None:
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        # only use the first num_context images as context and predict the rest
        images = tf.unstack(images, axis=1)
        context_images = []
        for i in range(num_context):
            context_images.append(images.pop(0))
        context_images = tf.stack(context_images, axis=1)
        images_to_predict = images
        # encode initial context
        context = self.encoder(context_images, 
                conditioning=[], num_rmb=encoder_depth)
        lstm_out = self.lstm(context)
        rgb = self.decoder(context_images, conditioning=lstm_out,
                num_rmb=decoder_depth, out_channels=out_channels)
        rgb = self.reshape_rgb(rgb, out_channels)
        predicted_images = self.get_intensities(rgb)
        predicted_images = tf.unstack(predicted_images, axis=1)
        # recursively predict image and feed back into encoder and lstm
        for i, inp in enumerate(images_to_predict):
            # empty image
            inp_shape = inp.get_shape().as_list()
            image = tf.zeros(inp.get_shape().as_list(), dtype=tf.float32)
            image = tf.expand_dims(image, axis=1)
            cond = tf.expand_dims(tf.unstack(lstm_out, axis=1)[-1], axis=1)
            # sequentially decode
            print('Unrolling pixel by pixel:')
            for i in trange(inp_shape[-3], desc='height'):
                for j in trange(inp_shape[-2], desc='width'):
                    for k in xrange(inp_shape[-1]): # channel
                        rgb = self.decoder(image, \
                                conditioning=cond, \
                                condition_first_image=True, \
                                num_rmb=decoder_depth, \
                                out_channels=out_channels,
                                disable_print=True)
                        rgb = self.reshape_rgb(rgb, out_channels)
                        predicted_image = self.get_intensities(rgb)
                        #image[:, i, j, k] = predicted_image[:, i, j, k] #TODO sess.run to avoid unrolling
                        image = predicted_image
            # use predicted image as context for subsequent images
            context = self.encoder(image, 
                    conditioning=[], num_rmb=encoder_depth)
            lstm_out = self.lstm(context)
            predicted_images.append(tf.squeeze(image))

        predicted_images = tf.stack(predicted_images, axis=1)
        return [{'predicted': predicted_images}, 
                {'encoder_depth': encoder_depth, 'decoder_depth': decoder_depth}]

def model(inputs,
        gaussian=None,
        stats_file=None,
        normalization_method=None,
        encoder_depth=8,
        decoder_depth=12,
        num_context=2,
        train=True,
        **kwargs):

    if normalization_method is not None:
        assert stats_file is not None, ('stats file has to be provided\
                to use normalization')
        normalization=Normalizer(stats_file, normalization_method)
    else:
        normalization=None

    net = VPN(inputs, gaussian=gaussian, normalization=normalization)
    if train:
        return net.train(encoder_depth, decoder_depth)
    else:
        return net.test(encoder_depth, decoder_depth)

def softmax_cross_entropy_loss(labels, logits, **kwargs):
    labels = tf.cast(labels, tf.int32)
    logits = logits['rgb']
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))

def parallel_model(inputs, n_gpus=4, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        assert n_gpus > 0, ('At least one gpu has to be used')
        outputs = []
        params = []
        inputs['images'] = tf.split(inputs['images'], axis=0, num_or_size_splits=n_gpus)
        for i, inp in enumerate(inputs['images']):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                    output, param = model({'images': inp}, **kwargs)
                    outputs.append(output)
                    params.append(param)
                    tf.get_variable_scope().reuse_variables()
        outputs = dict(zip(outputs[0],zip(*[d.values() for d in outputs])))
        params = params[0]
        return [outputs, params]

def parallel_softmax_cross_entropy_loss(labels, logits, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        logits = logits['rgb']
        n_gpus = len(logits)
        labels = tf.cast(labels, tf.int32)
        labels = tf.split(labels, axis=0, num_or_size_splits=n_gpus)
        losses = []
        for i, (label, logit) in enumerate(zip(labels, logits)):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                    label = tf.squeeze(label)
                    losses.append(
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=label, logits=logit)))
                    tf.get_variable_scope().reuse_variables()
        return losses

def parallel_reduce_mean(losses, **kwargs):
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        for i, loss in enumerate(losses):
            losses[i] = tf.reduce_mean(loss)
        return losses

class ParallelClipOptimizer(object):

    def __init__(self, optimizer_class, clip=True, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip

    def compute_gradients(self, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(*args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                   for grad, var in gvs if grad is not None]
        return gvs

    def minimize(self, losses, global_step):
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            grads_and_vars = []
            for i, loss in enumerate(losses):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('gpu_' + str(i)) as gpu_scope:
                        grads_and_vars.append(self.compute_gradients(loss))
                        #tf.get_variable_scope().reuse_variables()
            grads_and_vars = self.average_gradients(grads_and_vars)
            return self._optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)

    def average_gradients(self, all_grads_and_vars):
        average_grads_and_vars = []
        for grads_and_vars in zip(*all_grads_and_vars):
            grads = []
            for g, _ in grads_and_vars:
                grads.append(tf.expand_dims(g, axis=0))
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            # all variables are the same so we just use the first gpu variables
            var = grads_and_vars[0][1]
            grad_and_var = (grad, var)
            average_grads_and_vars.append(grad_and_var)
        return average_grads_and_vars
