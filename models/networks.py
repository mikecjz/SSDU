import tensorflow as tf
import numpy as np
import sys

from tensorflow.keras.layers import Conv2D, Lambda, Add, Activation


def conv_layer(input_data, conv_filter, is_relu=False, is_scaling=False):
    """
    Parameters
    ----------
    x : input data
    conv_filter : weights of the filter
    is_relu : applies  ReLU activation function
    is_scaling : Scales the output

    """

    n_filters = conv_filter[3]
    kernel = conv_filter[0:2]

    #setup activation type
    if (is_relu):
        activation_type = 'relu'
    else:
        activation_type = None

    #perform convolution
    x = Conv2D(n_filters, kernel, padding='same', activation=activation_type)(input_data)


    #optional scaling
    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = x * scalar

    return x


def ResNet(input_data, nb_res_blocks):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """

    conv_filters = dict([('w1', (3, 3, 2, 64)), ('w2', (3, 3, 64, 64)), ('w3', (3, 3, 64, 2))])
    intermediate_outputs = {}

    with tf.compat.v1.variable_scope('FirstLayer'):

        print('Shape of First Layer Input data: ' + str(input_data.shape))
        intermediate_outputs['layer0'] = conv_layer(input_data, conv_filters['w1'], is_relu=False, is_scaling=False)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.compat.v1.variable_scope('ResBlock' + str(i)):
            conv_layer1 = conv_layer(intermediate_outputs['layer' + str(i - 1)], conv_filters['w2'], is_relu=True, is_scaling=False)
            conv_layer2 = conv_layer(conv_layer1, conv_filters['w2'], is_relu=False, is_scaling=True)

            intermediate_outputs['layer' + str(i)] = conv_layer2 + intermediate_outputs['layer' + str(i - 1)]

    with tf.compat.v1.variable_scope('LastLayer'):
        rb_output = conv_layer(intermediate_outputs['layer' + str(i)], conv_filters['w2'], is_relu=False, is_scaling=False)

    with tf.compat.v1.variable_scope('Residual'):
        temp_output = rb_output + intermediate_outputs['layer0']
        nw_output = conv_layer(temp_output, conv_filters['w3'], is_relu=False, is_scaling=False)

    return nw_output


def mu_param():
    """
    Penalty parameter used in DC units, x = (E^h E + \mu I)^-1 (E^h y + \mu * z)
    """

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        mu = tf.compat.v1.get_variable(name='mu', dtype=tf.float32, initializer=.05)

    return mu
