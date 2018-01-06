import tensorflow as tf


def convolution_2d(name, label, var_in, f, dim_in, dim_out, initializer, transfer, reuse=False):
    """Standard convolutional layer"""
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope(label, reuse=reuse):
            if reuse:
                W = tf.get_variable("W", [f, f, dim_in, dim_out])
                b = tf.get_variable("b", [dim_out])
            else: # new
                W = tf.get_variable("W", [f, f, dim_in, dim_out], initializer=initializer)
                b = tf.get_variable("b", [dim_out], initializer=initializer)

    z_hat = tf.nn.conv2d(var_in, W, strides=[1,1,1,1], padding="SAME")
    z_hat = tf.nn.bias_add(z_hat, b)
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