import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
Critic models takes two inputs:
    1. encoded states
    2. actions
and returns an expected future reward (a single float).

Critic model functions should take a following argument:
    1. encoded_state_shape
    2. action_space : Box
"""


def critic_simple_dense(encoded_state_shape, action_space):
    action_input = keras.Input(action_space.shape,
                            name='Action_input')
    encoded_state_input = keras.Input(encoded_state_shape)
    s = layers.Flatten(name='critic_flatten_state')(encoded_state_input)
    a = layers.Flatten(name='critic_flatten_action')(action_input)

    x = layers.Concatenate(name='critic_concat_action_state')([s,a])

    x = layers.Dense(256, activation='relu',
                     name='critic_dense1')(x)
    x = layers.Dense(128, activation='relu',
                     name='critic_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='critic_dense3')(x)
    x = layers.Dense(1, activation='linear',dtype='float32',
                           name='critic_dense4')(x)
    outputs = tf.squeeze(x, name='critic_squeeze')

    model = keras.Model(
        inputs=[action_input, encoded_state_input], 
        outputs=outputs,
        name='critic'
    )

    return model
