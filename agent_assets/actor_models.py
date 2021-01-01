import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
Actor models takes one input:
    1. encoded states
and returns an action.

Critic model functions should take a following argument:
    1. encoded_state_shape
    2. action_space : Box
"""

def actor_simple_dense(encoded_state_shape, action_space):
    encoded_state_input = keras.Input(encoded_state_shape)
    s = layers.Flatten(name='actor_flatten_state')(encoded_state_input)
    action_shape = action_space.shape
    action_num = tf.math.reduce_prod(action_shape)
    action_range = action_space.high - action_space.low
    action_low = action_space.low

    x = layers.Dense(256, activation='relu',
                     name='actor_dense1')(s)
    x = layers.Dense(128, activation='relu',
                     name='actor_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='actor_dense3')(x)
    x = layers.Dense(action_num, activation='hard_sigmoid',dtype='float32',
                           name='actor_dense4')(x)
    x = layers.Reshape(action_space.shape, name='actor_reshape')(x)
    outputs = x*action_range + action_low

    model = keras.Model(
        inputs=encoded_state_input,
        outputs=outputs,
        name='actor'
    )

    return model