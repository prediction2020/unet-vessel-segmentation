"""
File name: unet.py
Author: Jana Rieger
Date created: 02/24/2018

This file defines the unet architecture.
"""

from keras.models import Model
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Input, UpSampling2D, concatenate, BatchNormalization


def conv_block(m, num_kernels, kernel_size, strides, padding, activation, dropout, data_format, bn):
    """
    Bulding block with convolutional layers for one level.

    :param m: model
    :param num_kernels: number of convolution filters on the particular level, positive integer
    :param kernel_size: size of the convolution kernel, tuple of two positive integers
    :param strides: strides values, tuple of two positive integers
    :param padding: used padding by convolution, takes values: 'same' or 'valid'
    :param activation: activation_function after every convolution
    :param dropout: percentage of weights to be dropped, float between 0 and 1
    :param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
    :param bn: weather to use Batch Normalization layers after each convolution layer, True for use Batch Normalization,
     False do not use Batch Normalization
    :return: model
    """
    n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
                      data_format=data_format)(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(dropout)(n)
    n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
                      data_format=data_format)(n)
    n = BatchNormalization()(n) if bn else n
    return n


def up_concat_block(m, concat_channels, pool_size, concat_axis, data_format):
    """
    Bulding block with up-sampling and concatenation for one level.

    :param m: model
    :param concat_channels: channels from left side onf Unet to be concatenated with the right part on one level
    :param pool_size: factors by which to downscale (vertical, horizontal), tuple of two positive integers
    :param concat_axis: concatenation axis, concatenate over channels, positive integer
    :param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
    :return: model    """
    n = UpSampling2D(size=pool_size, data_format=data_format)(m)
    n = concatenate([n, concat_channels], axis=concat_axis)
    return n


def get_unet(patch_size, num_channels, activation, final_activation, optimizer, learning_rate, dropout,
             loss_function, metrics=None,
             kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=3,
             data_format='channels_last', padding='same', bn=False):
    """
    Defines the architecture of the u-net. Reconstruction of the u-net introduced in: https://arxiv.org/abs/1505.04597

    :param patch_size: height of the patches, positive integer
    :param num_channels: number of channels of the input images, positive integer
    :param activation: activation_function after every convolution
    :param final_activation: activation_function of the final layer
    :param optimizer: optimization algorithm for updating the weights and bias values
    :param learning_rate: learning_rate of the optimizer, float
    :param dropout: percentage of weights to be dropped, float between 0 and 1
    :param loss_function: loss function also known as cost function
    :param metrics: metrics for evaluation of the model performance
    :param kernel_size: size of the convolution kernel, tuple of two positive integers
    :param pool_size: factors by which to downscale (vertical, horizontal), tuple of two positive integers
    :param strides: strides values, tuple of two positive integers
    :param num_kernels: array specifying the number of convolution filters in every level, list of positive integers
        containing value for each level of the model
    :param concat_axis: concatenation axis, concatenate over channels, positive integer
    :param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
    :param padding: used padding by convolution, takes values: 'same' or 'valid'
    :param bn: weather to use Batch Normalization layers after each convolution layer, True for use Batch Normalization,
     False do not use Batch Normalization
    :return: compiled u-net model
    """
    if metrics is None:
        metrics = ['accuracy']
    if num_kernels is None:
        num_kernels = [64, 128, 256, 512, 1024]

    # specify the input shape
    inputs = Input((patch_size, patch_size, num_channels))

    # DOWN-SAMPLING PART (left side of the U-net)
    # layers on each level: convolution2d -> dropout -> convolution2d -> max-pooling
    # last level without max-pooling

    # level 0
    conv_0_down = conv_block(inputs, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_0 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_0_down)

    # level 1
    conv_1_down = conv_block(pool_0, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_1 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_1_down)

    # level 2
    conv_2_down = conv_block(pool_1, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_2 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_2_down)

    # level 3
    conv_3_down = conv_block(pool_2, num_kernels[3], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_3 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_3_down)

    # level 4
    conv_4 = conv_block(pool_3, num_kernels[4], kernel_size, strides, padding, activation, dropout, data_format, bn)

    # UP-SAMPLING PART (right side of the U-net)
    # layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
    # part -> convolution2d -> dropout -> convolution2d
    # final convolutional layer maps feature maps to desired number of classes

    # level 3
    concat_3 = up_concat_block(conv_4, conv_3_down, pool_size, concat_axis, data_format)
    conv_3_up = conv_block(concat_3, num_kernels[3], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)

    # level 2
    concat_2 = up_concat_block(conv_3_up, conv_2_down, pool_size, concat_axis, data_format)
    conv_2_up = conv_block(concat_2, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)

    # level 1
    concat_1 = up_concat_block(conv_2_up, conv_1_down, pool_size, concat_axis, data_format)
    conv_1_up = conv_block(concat_1, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)

    # level 0
    concat_0 = up_concat_block(conv_1_up, conv_0_down, pool_size, concat_axis, data_format)
    conv_0_up = conv_block(concat_0, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)
    final_conv = Convolution2D(1, 1, strides=strides, activation=final_activation, padding=padding,
                               data_format=data_format)(conv_0_up)

    # configure the learning process via the compile function
    model = Model(inputs=inputs, outputs=final_conv)
    model.compile(optimizer=optimizer(lr=learning_rate), loss=loss_function,
                  metrics=metrics)
    print('U-net compiled.')

    # print out model summary to console
    model.summary()

    return model
