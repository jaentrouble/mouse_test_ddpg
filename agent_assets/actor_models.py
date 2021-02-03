import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
"""
Actor models takes one input:
    1. encoded states
and returns an action.

Critic model functions should take following arguments:
    1. observation_space
    2. action_space : Box
    3. encoder_f
"""
EPSILON = 1e-10

def actor_simple_dense(observation_space, action_space, encoder_f):
    encoded_state, encoder_inputs = encoder_f(observation_space)
    s = layers.Flatten(name='actor_flatten_state')(encoded_state)
    action_shape = action_space.shape
    action_num = tf.math.reduce_prod(action_shape)
    action_range = action_space.high - action_space.low
    action_middle = (action_space.low + action_space.high)/2

    x = layers.Dense(256, activation='relu',
                     name='actor_dense1')(s)
    x = layers.Dense(128, activation='relu',
                     name='actor_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='actor_dense3')(x)
    x = layers.Dense(action_num, activation='tanh',
                     name='actor_dense4',)(x)
    x = layers.Reshape(action_space.shape, name='actor_reshape')(x)
    outputs = x*action_range/2 + action_middle
    outputs = layers.Activation('linear',dtype='float32',
                                         name='actor_float32')(outputs)

    model = keras.Model(
        inputs=encoder_inputs,
        outputs=outputs,
        name='actor'
    )

    return model

def actor_soft_dense(observation_space, action_space, encoder_f):
    """
    returns action and log_pi
    """
    encoded_state, encoder_inputs = encoder_f(observation_space)
    s = layers.Flatten(name='actor_flatten_state')(encoded_state)
    batch_size = tf.shape(s)[0]

    action_shape = action_space.shape
    action_num = tf.math.reduce_prod(action_shape)
    action_range = action_space.high - action_space.low
    action_middle = (action_space.low + action_space.high)/2

    x = layers.Dense(256, activation='relu',
                     name='actor_dense1')(s)
    x = layers.Dense(128, activation='relu',
                     name='actor_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='actor_dense3')(x)
    mu = layers.Dense(action_num, activation='linear', dtype='float32',
                     name='actor_mu',)(x)
    log_sigma = layers.Dense(action_num, activation='linear',
                             dtype='float32',name='actor_logsig')(x)
    sigma = tf.exp(log_sigma)

    normal_dist = tfp.distributions.MultivariateNormalDiag(
        mu, sigma, name='actor_dist')
    raw_action = normal_dist.sample(batch_size)

    squashed_action = keras.activations.tanh(raw_action)

    raw_log_pi = normal_dist.log_prob(raw_action)
    log_pi = raw_log_pi - tf.reduce_sum(
        tf.math.log(1 - squashed_action**2 + EPSILON),
        axis=-1,
    )

    reshaped_action = layers.Reshape(
        action_space.shape, name='actor_reshape'
    )(squashed_action)
    actions = reshaped_action*action_range/2 + action_middle
    actions = layers.Activation('linear',dtype='float32',
                                         name='actor_float32')(actions)

    model = keras.Model(
        inputs=encoder_inputs,
        outputs=[actions, log_pi],
        name='actor'
    )

    return model