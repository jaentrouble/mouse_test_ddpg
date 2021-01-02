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


def critic_simple_dense(observation_space, action_space, encoder_f):
    action_input = keras.Input(action_space.shape,
                            name='action')
    encoded_state, encoder_inputs = encoder_f(observation_space)
    s = layers.Flatten(name='critic_flatten_state')(encoded_state)
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
        inputs=[action_input,] + encoder_inputs, 
        outputs=outputs,
        name='critic'
    )

    return model
