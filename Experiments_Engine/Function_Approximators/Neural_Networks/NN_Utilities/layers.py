import tensorflow as tf


def convolution_2d(name, label, var_in, f, dim_in, dim_out, initializer, transfer, reuse=False, stride=1, format='NCHW'):
    """Standard convolutional layer"""
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope(label, reuse=reuse):
            if reuse:
                W = tf.get_variable("W", [f, f, dim_in, dim_out])
                b = tf.get_variable("b", [dim_out])
            else: # new
                W = tf.get_variable("W", [f, f, dim_in, dim_out], initializer=initializer)
                b = tf.get_variable("b", [dim_out], initializer=initializer)
    strides = [1,1,stride,stride]
    if format == 'NHWC':
        strides = [1,stride, stride, 1]
    z_hat = tf.nn.conv2d(var_in, W, strides=strides, padding="SAME", data_format=format)
    z_hat = tf.nn.bias_add(z_hat, b, data_format=format)
    y_hat = transfer(z_hat)
    return W, b, z_hat, y_hat


def fully_connected(name, label, var_in, dim_in, dim_out, initializer, transfer, reuse=False):
    """Standard fully connected layer"""
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope(label, reuse=reuse):
            if reuse:
                W = tf.get_variable("W", [dim_in, dim_out])
                b = tf.get_variable("b", [dim_out])
            else: # new
                W = tf.get_variable("W", [dim_in, dim_out], initializer=initializer)
                b = tf.get_variable("b", [dim_out], initializer=initializer)

    z_hat = tf.matmul(var_in, W)
    z_hat = tf.nn.bias_add(z_hat, b)
    y_hat = transfer(z_hat)
    return W, b, z_hat, y_hat


def fully_connected_rbf(name, label, var_in, dim_in, dim_out, initializer, center, stddev, reuse=False):
    """Standard fully connected layer"""
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope(label, reuse=reuse):
            if reuse:
                W = tf.get_variable("W", [dim_in, dim_out])
                b = tf.get_variable("b", [dim_out])
            else: # new
                W = tf.get_variable("W", [dim_in, dim_out], initializer=initializer)
                b = tf.get_variable("b", [dim_out], initializer=initializer)

    z_hat = tf.matmul(var_in, W)
    z_hat = tf.nn.bias_add(z_hat, b)
    max_z_hat = tf.reduce_max(z_hat)
    z_hat = tf.divide(z_hat, max_z_hat)
    centered = tf.subtract(z_hat, center)
    y_hat = tf.exp(tf.multiply(tf.divide(tf.pow(centered,2), stddev), -1))
    return W, b, z_hat, y_hat

def convolution_2d_rbf(name, label, var_in, f, dim_in, dim_out, initializer, center,stddev, reuse=False, stride=1,
                       format='NCHW'):
    """Standard convolutional layer"""
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope(label, reuse=reuse):
            if reuse:
                W = tf.get_variable("W", [f, f, dim_in, dim_out])
                b = tf.get_variable("b", [dim_out])
            else: # new
                W = tf.get_variable("W", [f, f, dim_in, dim_out], initializer=initializer)
                b = tf.get_variable("b", [dim_out], initializer=initializer)
    strides = [1,1,stride,stride]
    if format == 'NHWC':
        strides = [1,stride, stride, 1]
    z_hat = tf.nn.conv2d(var_in, W, strides=strides, padding="SAME", data_format=format)
    z_hat = tf.nn.bias_add(z_hat, b, data_format=format)
    max_z_hat = tf.reduce_max(z_hat)
    z_hat = tf.divide(z_hat, max_z_hat)
    centered = tf.subtract(z_hat, center)
    y_hat = tf.exp(tf.multiply(tf.divide(tf.pow(centered,2), stddev), -1))
    return W, b, z_hat, y_hat
