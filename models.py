import tensorflow as tf

class RBFActivation(tf.keras.layers.Layer):
    """Defines RBF layer.

    This function defines a RBF layer from [1].

    [1] “Gradient-Based Learning Applied to Document Recognition”,
      Y. LeCun, L. Bottou, Y. Bengio and P. Haffner (1998).
    """

    def __init__(self, units):
        super(RBFActivation, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        input_dim = input_shape[1]
        weight_shape = (input_dim._value, self.units)
        self.kernel = self.add_weight('kernel',
                                      shape=weight_shape,
                                      trainable=True)
        super().build(input_shape)
    
    def call(self, inputs):
        res = tf.reduce_sum(
            tf.square(inputs[..., None] - self.kernel), axis=1)
        return res

class AvgPoolWithWeights(tf.keras.layers.AveragePooling2D):
    """Average pooling with learnable coefficients.

    This function defines an average pooling from [1]. This result
    of `tf.keras.layers.AveragePooling2D` is multiplied by a learnable
    coefficient (one per map) and adds a learnable bias term (one
    per map) and applies activation function (if not `None`).

    [1] “Gradient-Based Learning Applied to Document Recognition”,
      Y. LeCun, L. Bottou, Y. Bengio and P. Haffner (1998).
    """
    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 data_format=None,
                 activation=None,
                 **kwargs):
        
        super(AvgPoolWithWeights, self).__init__(pool_size=pool_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 data_format=data_format,
                                                 **kwargs)
        self.activation = activation

    def build(self, input_shape):
        output_shape = self.compute_output_shape(input_shape)
        weight_shape = output_shape[-1:]
        self.kernel = self.add_weight(name='kernel',
                                      shape=weight_shape,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=weight_shape,
                                    initializer=tf.initializers.zeros,
                                    trainable=True)
        super(AvgPoolWithWeights, self).build(input_shape)

    def call(self, inputs, **kwargs):
        res = super().call(inputs, **kwargs)
        res = self.kernel * res + self.bias
        if self.activation is not None:
            return self.activation(res)
        return res


def build_lenet5(inputs, dropout_rate, n_outputs=10):
    
    with tf.name_scope('pad-inputs'):
        res = tf.keras.layers.Lambda(
            lambda x: tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]]))(inputs)

    with tf.name_scope('conv1'):
        res = tf.keras.layers.Conv2D(filters=6,
                                     kernel_size=5,
                                     strides=(1, 1),
                                     padding='VALID',
                                     activation=tf.nn.tanh)(res)

    with tf.name_scope('avg-pool-with-weights1'):
        res = AvgPoolWithWeights(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='valid',
                                 activation=tf.nn.tanh)(res)

    with tf.name_scope('dropout1'):
        res = tf.keras.layers.Dropout(dropout_rate)(res)
        
    with tf.name_scope('conv2'):
        res = tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=5,
                                     padding='valid',
                                     activation=tf.nn.tanh)(res)
    
    with tf.name_scope('avg-pool-with-weights2'):
        res = AvgPoolWithWeights(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='valid',
                                 activation=tf.nn.tanh)(res)

    with tf.name_scope('dropout2'):
        res = tf.keras.layers.Dropout(dropout_rate)(res)
    
    with tf.name_scope('conv3'):
        res = tf.keras.layers.Conv2D(filters=120,
                                     kernel_size=5,
                                     padding='valid',
                                     activation=tf.nn.tanh)(res)
    with tf.name_scope('dropout3'):
        res = tf.keras.layers.Dropout(dropout_rate)(res)
    
    with tf.name_scope('fc1'):
        res = tf.keras.layers.Flatten()(res)
        res = tf.keras.layers.Dense(units=84,
                                    activation=tf.nn.tanh)(res)
    
    with tf.name_scope('logits'):
        outputs = RBFActivation(units=n_outputs)(res)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

def build_simple_conv_net(inputs, dropout_rate, n_outputs=10):
    
    with tf.name_scope('pad-inputs'):
        res = tf.keras.layers.Lambda(
            lambda x: tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]]))(inputs)

    with tf.name_scope('conv1'):
        res = tf.keras.layers.Conv2D(filters=6,
                                     kernel_size=5,
                                     strides=(1, 1),
                                     padding='VALID',
                                     activation=tf.nn.tanh)(res)

    with tf.name_scope('avg-pool-with-weights1'):
        res = AvgPoolWithWeights(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='valid',
                                 activation=tf.nn.tanh)(res)

    with tf.name_scope('dropout1'):
        res = tf.keras.layers.Dropout(dropout_rate)(res)
        
    with tf.name_scope('conv2'):
        res = tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=5,
                                     padding='valid',
                                     activation=tf.nn.tanh)(res)
    
    with tf.name_scope('avg-pool-with-weights2'):
        res = AvgPoolWithWeights(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='valid',
                                 activation=tf.nn.tanh)(res)

    with tf.name_scope('dropout2'):
        res = tf.keras.layers.Dropout(dropout_rate)(res)
    
    
    with tf.name_scope('logits'):
        res = tf.keras.layers.Flatten()(res)
        res = tf.keras.layers.Dense(units=n_outputs,
                                    activation=None)(res)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[res])
    return model