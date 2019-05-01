#configuration fo convolution networks of large and small

import tensorflow as tf

def build_convolution(name,input,case,train_phase):
    if case=='small':
        conv = tf.nn.selu(tf.layers.conv2d( \
            inputs=input, filters=32, \
            kernel_size=[5, 5], strides=[2, 2], activation=None, reuse=None, padding='VALID', \
            name=name+ '_net_conv2'))
        # conv = tf.layers.batch_normalization(conv, training=train_phase,trainable=True)
        conv = tf.nn.selu(tf.layers.conv2d( \
            inputs=conv, filters=64, \
            kernel_size=[3, 3], strides=[1, 1], activation=None, reuse=None, padding='VALID', \
            name=name+ '_net_conv3'))
        # conv = tf.layers.batch_normalization(conv, training=train_phase, trainable=True)
        return conv

    if case=='large':
        conv =tf.nn.selu(tf.layers.conv2d( \
            inputs=input, filters=16, \
            kernel_size=[5, 5], strides=[3, 3], activation=None,reuse=None,padding='VALID', \
            name=name + '_net_conv1'))
        conv =tf.nn.selu(tf.layers.conv2d( \
            inputs=conv, filters=32, \
            kernel_size=[3, 3], strides=[2, 2], activation=None,reuse=None,padding='VALID', \
            name=name + '_net_conv2'))
        conv = tf.nn.selu(tf.layers.conv2d( \
            inputs=conv, filters=64, \
            kernel_size=[3, 3], strides=[1, 1], activation=None,reuse=None,padding='VALID', \
            name=name + '_net_conv3'))

        return conv