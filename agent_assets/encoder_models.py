import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
Encoder models encode states into a feature tensor.

Encoder model functions should take a following argument:
    1. observation_space : Dict
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
                            name='Right_eye_input')
    left_input = keras.Input(observation_space['Left'].shape,
                            name='Left_eye_input')

    right_encoded = single_eye(right_input, 'right')
    left_encoded = single_eye(left_input, 'left')

    concat_eyes = layers.Concatenate(
        name='encoder_concat_eyes'
    )([left_encoded, right_encoded])

    x = layers.Flatten(name='encoder_flatten')(concat_eyes)
    outputs = layers.Dense(256, activation='linear',
                     name='encoder_dense')(x)

    model = keras.Model(
        inputs=[right_input, left_input],
        outputs=outputs,
        name='encoder'
    )
    return model