import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
Actor models takes one input:
    1. encoded states
and returns an action.

Critic model functions should take following arguments:
    1. observation_space
    2. action_space : Box
    3. encoder_f
"""

def actor_simple_dense(observation_space, action_space, encoder_f):
    encoded_state, encoder_inputs = encoder_f(observation_space)
    s = layers.Flatten(name='actor_flatten_state')(encoded_state)
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
    x = layers.Dense(action_num, activation='linear',
                     name='actor_dense4')(x)
    x = layers.Reshape(action_space.shape, name='actor_reshape')(x)
    outputs = x*action_range + action_low
    outputs = layers.Activation('linear',dtype='float32',
                                         name='actor_float32')(outputs)

    model = keras.Model(
        inputs=encoder_inputs,
        outputs=outputs,
        name='actor'
    )

    return model