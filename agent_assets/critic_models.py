import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
Critic models takes two inputs:
    1. states (observations)
    2. actions
and returns an expected future reward (a single float).

Critic model functions should take folllowing two as inputs:
    1. observation_space : Dict
    2. action_space : Box, expected to be 1-dimension
"""

def eye_model(inputs, left_or_right):
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
    x = layers.Conv1D(128, 3, strides=2, activation='relu'
                      name=left_or_right+'_eye_conv2')(x)
    outputs = layers.Conv1D(256, 3, strides=2, activation='relu',
                      name=left_or_right+'_eye_conv3')(x)
    return outputs

def brain_layers(encoded_state, action):
    s = layers.Flatten(name='flatten_state')(encoded_state)
    a = layers.Flatten(name='flatten_action')(action)

    x = layers.Concatenate(name='brain_concat_action_state')([s,a])

    x = layers.Dense(256, activation='relu',
                     name='brain_dense1')(x)
    x = layers.Dense(128, activation='relu'
                     name='brain_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='brain_dense3')(x)
    x = layers.Dense(1, activation='linear',dtype='float32',
                           name='brain_dense4')(x)
    outputs = tf.squeeze(x, name='brain_squeeze')
    return outputs

def eye_brain_model(observation_space, action_space):
    right_input = keras.Input(observation_space['Right'].shape,
                            name='Right_eye_input')
    left_input = keras.Input(observation_space['Left'].shape,
                            name='Left_eye_input')
    action_input = keras.Input(action_space.shape,
                            name='Action_input')

    right_encoded = eye_model(right_input, 'right')
    left_encoded = eye_model(left_input, 'left')

    encoded_state = layers.Concatenate(
        name='concat_left_right_eyes'
    )([left_encoded, right_encoded])
    
    outputs = brain_layers(encoded_state,action_input)

    model = keras.Model(
        inputs=[left_input, right_input, action_input], 
        outputs=outputs,
        name='eye_brain_critic'
    )

    return model
