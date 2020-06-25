import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow_compression as tfc
import resnet

def resblock(input, IC, OC, name):

    l1 = tf.nn.relu(input, name=name + 'relu1')

    l1 = tf.layers.conv2d(inputs=l1, filters=np.minimum(IC, OC), kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l1')

    l2 = tf.nn.relu(l1, name='relu2')

    l2 = tf.layers.conv2d(inputs=l2, filters=OC, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l2')

    if IC != OC:
        input = tf.layers.conv2d(inputs=input, filters=OC, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'map')

    return input + l2


def MC_new(input):

    m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

    m2 = resblock(m1, 64, 64, name='mc2')

    m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

    m4 = resblock(m3, 64, 64, name='mc4')

    m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

    m6 = resblock(m5, 64, 64, name='mc6')

    m7 = resblock(m6, 64, 64, name='mc7')

    m8 = tf.image.resize_images(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

    m8 = m4 + m8

    m9 = resblock(m8, 64, 64, name='mc9')

    m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

    m10 = m2 + m10

    m11 = resblock(m10, 64, 64, name='mc11')

    m12 = tf.layers.conv2d(inputs=m11, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

    m12 = tf.nn.relu(m12, name='relu12')

    m13 = tf.layers.conv2d(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13')

    return m13

def MC(input, is_training, norm):

    m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')
    if norm == True:
        m1 = tf.layers.batch_normalization(inputs=m1, training=is_training, name='bn1')
    m1 = tf.nn.leaky_relu(m1, name='relu1')

    m2_1 = tf.layers.conv2d(inputs=m1, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc2_1')
    if norm == True:
        m2_1 = tf.layers.batch_normalization(inputs=m2_1, training=is_training, name='bn2_1')
    m2_1 = tf.nn.leaky_relu(m2_1, name='relu2_1')

    m2_2 = tf.layers.conv2d(inputs=m2_1, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc2_2')
    if norm == True:
        m2_2 = tf.layers.batch_normalization(inputs=m2_2, training=is_training, name='bn2_2')
    m2 = tf.nn.leaky_relu(m2_2 + m1, name='relu2_2')

    m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

    m4_1 = tf.layers.conv2d(inputs=m3, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc4_1')
    if norm == True:
        m4_1 = tf.layers.batch_normalization(inputs=m4_1, training=is_training, name='bn4_1')
    m4_1 = tf.nn.leaky_relu(m4_1, name='relu4_1')

    m4_2 = tf.layers.conv2d(inputs=m4_1, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc4_2')
    if norm == True:
        m4_2 = tf.layers.batch_normalization(inputs=m4_2, training=is_training, name='bn4_2')
    m4 = tf.nn.leaky_relu(m4_2 + m3, name='relu4_2')

    m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

    m6_1 = tf.layers.conv2d(inputs=m5, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc6_1')
    if norm == True:
        m6_1 = tf.layers.batch_normalization(inputs=m6_1, training=is_training, name='bn6_1')
    m6_1 = tf.nn.leaky_relu(m6_1, name='relu6_1')

    m6_2 = tf.layers.conv2d(inputs=m6_1, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc6_2')
    if norm == True:
        m6_2 = tf.layers.batch_normalization(inputs=m6_2, training=is_training, name='bn6_2')
    m6 = tf.nn.leaky_relu(m6_2 + m5, name='relu6_2')

    m7_1 = tf.layers.conv2d(inputs=m6, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc7_1')
    if norm == True:
        m7_1 = tf.layers.batch_normalization(inputs=m7_1, training=is_training, name='bn7_1')
    m7_1 = tf.nn.leaky_relu(m7_1, name='relu7_1')

    m7_2 = tf.layers.conv2d(inputs=m7_1, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc7_2')
    if norm == True:
        m7_2 = tf.layers.batch_normalization(inputs=m7_2, training=is_training, name='bn7_2')
    m7 = tf.nn.leaky_relu(m7_2 + m6, name='relu7_2')

    m8 = tf.image.resize_images(m7, [2*tf.shape(m7)[1], 2*tf.shape(m7)[2]])
    m8 = tf.concat([m4, m8], axis=-1)
    m8 = tf.layers.conv2d(inputs=m8, filters=64, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc8')
    if norm == True:
        m8 = tf.layers.batch_normalization(inputs=m8, training=is_training, name='bn8')

    m9_1 = tf.layers.conv2d(inputs=m8, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc9_1')
    if norm == True:
        m9_1 = tf.layers.batch_normalization(inputs=m9_1, training=is_training, name='bn9_1')
    m9_1 = tf.nn.leaky_relu(m9_1, name='relu9_1')

    m9_2 = tf.layers.conv2d(inputs=m9_1, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc9_2')
    if norm == True:
        m9_2 = tf.layers.batch_normalization(inputs=m9_2, training=is_training, name='bn9_2')
    m9 = tf.nn.leaky_relu(m9_2 + m8, name='relu9_2')

    m10 = tf.image.resize_images(m9, [2*tf.shape(m9)[1], 2*tf.shape(m9)[2]])
    m10 = tf.concat([m2, m10], axis=-1)
    m10 = tf.layers.conv2d(inputs=m10, filters=64, kernel_size=1, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc10')
    if norm == True:
        m10 = tf.layers.batch_normalization(inputs=m10, training=is_training, name='bn10')

    m11_1 = tf.layers.conv2d(inputs=m10, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc11_1')
    if norm == True:
        m11_1 = tf.layers.batch_normalization(inputs=m11_1, training=is_training, name='bn11_1')
    m11_1 = tf.nn.leaky_relu(m11_1, name='relu11_1')

    m11_2 = tf.layers.conv2d(inputs=m11_1, filters=64, kernel_size=3, strides=1, padding='same',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc11_2')
    if norm == True:
        m11_2 = tf.layers.batch_normalization(inputs=m11_2, training=is_training, name='bn11_2')
    m11 = tf.nn.leaky_relu(m11_2 + m10, name='relu11_2')

    m12 = tf.layers.conv2d(inputs=m11, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')
    if norm == True:
        m12 = tf.layers.batch_normalization(inputs=m12, training=is_training, name='bn12')
    m12 = tf.nn.leaky_relu(m12, name='relu12')

    m13 = tf.layers.conv2d(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13')

    return m13