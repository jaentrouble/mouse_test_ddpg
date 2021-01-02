import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import critic_models as cm
import encoder_models as em
import actor_models as am
"""
Actor-Critic agent model
Agent functions return two models:
    1. encoder_model
        This takes observation only
    2. actor_model
        This takes encoded state only
    3. critic_model
        This takes encoded state and action together

Every functions should take following two as inputs:
    1. observation_space
    2. action_space : Box expected
"""

def eye_brain_model(observation_space, action_space):
    
    encoder = em.encoder_two_eyes(observation_space)
    encoded_state_shape = encoder.output.shape[1:]

    actor = am.actor_simple_dense(encoded_state_shape, action_space)
    
    critic = cm.critic_simple_dense(encoded_state_shape, action_space)

    return encoder, actor, critic

def mountaincar_model(observation_space, action_space):
    encoder = em.encoder_simple_dense(observation_space)
    encoded_state_shape = encoder.output.shape[1:]

    actor = am.actor_simple_dense(encoded_state_shape, action_space)

    critic = cm.critic_simple_dense(encoded_state_shape, action_space)

    return encoder, actor, critic

if __name__ == '__main__':
    from gym.spaces import Dict, Box
    import numpy as np
    observation_space = Dict(
        {'Right' : Box(0, 255, shape=(100,3,3), dtype=np.uint8),
         'Left' : Box(0,255, shape=(100,3,3), dtype = np.uint8)}
    )
    action_space = Box(
        low=np.array([-10.0,-np.pi]),
        high=np.array([10.0,np.pi]),
        dtype=np.float32
    )
    encoder, actor, critic = eye_brain_model(observation_space,action_space)
    encoder.summary()
    actor.summary()
    critic.summary()
