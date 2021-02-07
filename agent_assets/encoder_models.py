import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .nn_tools import *
"""
Encoder models encode states into a feature tensor.

Encoder model functions should take a following argument:
    1. observation_space : Dict
Encoder model functions should return:
    1. output tensor
    2. list of Inputs
"""

def single_eye(inputs, left_or_right):
    """
    Return an eye model
    Parameters
    ----------
    inputs : keras.Input

    left_or_right : str
    """
    x = layers.Reshape((inputs.shape[1],
                        inputs.shape[2]*inputs.shape[3]))(inputs)
    x = layers.Conv1D(64, 3, strides=1, activation='relu',
                      name=left_or_right+'_eye_conv1')(x)
    x = layers.Conv1D(128, 3, strides=2, activation='relu',
                      name=left_or_right+'_eye_conv2')(x)
    x = layers.Conv1D(256, 3, strides=2, activation='relu',
                      name=left_or_right+'_eye_conv3')(x)
    outputs = layers.GlobalMaxPool1D(name=left_or_right+'_eye_max_pooling')(x)
    return outputs

def encoder_two_eyes(observation_space):
    right_input = keras.Input(observation_space['Right'].shape,
                            name='Right')
    left_input = keras.Input(observation_space['Left'].shape,
                            name='Left')

    right_encoded = single_eye(right_input, 'right')
    left_encoded = single_eye(left_input, 'left')

    concat_eyes = layers.Concatenate(
        name='encoder_concat_eyes'
    )([left_encoded, right_encoded])

    x = layers.Flatten(name='encoder_flatten')(concat_eyes)
    outputs = layers.Dense(256, activation='linear',
                     name='encoder_dense')(x)

    return outputs, [right_input, left_input]

def encoder_simple_dense(observation_space):
    inputs = keras.Input(observation_space['obs'].shape,
                         name='obs')
    x = layers.Flatten(name='encoder_flatten')(inputs)
    x = layers.Dense(256, activation='relu',
                         name='encoder_dense1')(x)
    outputs = layers.Dense(256, activation='linear',
                         name='encoder_dense2')(x)
    return outputs, [inputs]

def encoder_simple_res(observation_space):
    inputs = keras.Input(observation_space['obs'].shape,
                         name='obs')
    x = layers.Conv2D(
        32, 
        3, 
        padding='same',
        activation='relu',
        name='encoder_conv1'
    )(inputs)
    x = res_block(x, 2, name='encoder_resblock1')
    x = layers.Conv2D(
        64,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck1'
    )(x)
    x = res_block(x, 2, name='encoder_resblock2')
    x = layers.Conv2D(
        128,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck2'
    )(x)
    x = res_block(x, 2, name='encoder_resblock3')
    x = layers.Conv2D(
        256,
        3,
        padding='same',
        strides=2,
        name='encoder_bottleneck3'
    )(x)
    outputs = layers.GlobalMaxPool2D(name='encoder_pool',dtype='float32')(x)
    return outputs, [inputs]

def encoder_simple_conv(observation_space):
    inputs = keras.Input(observation_space['obs'].shape,
                         name='obs')
    x = layers.Conv2D(
        32, 
        3, 
        padding='same',
        activation='relu',
        name='encoder_conv1'
    )(inputs)
    x = layers.Conv2D(
        64,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck1'
    )(x)
    x = layers.Conv2D(
        128,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck2'
    )(x)
    x = layers.Conv2D(
        256,
        3,
        padding='same',
        strides=2,
        name='encoder_bottleneck3'
    )(x)
    x = layers.Conv2D(
        512,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck4'
    )(x)
    x = layers.Flatten(name='encoder_flatten')(x)
    outputs = layers.Dense(
        256,
        activation='linear',
        name='encoder_Dense'
    )(x)
    return outputs, [inputs]
