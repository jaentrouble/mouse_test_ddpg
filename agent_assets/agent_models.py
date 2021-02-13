import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from . import critic_models as cm
from . import encoder_models as em
from . import actor_models as am
from . import ICM
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
    
    encoder_f = em.encoder_two_eyes

    actor = am.actor_simple_dense(observation_space, action_space, encoder_f)
    
    critic = cm.critic_simple_dense(observation_space, action_space, encoder_f)

    return actor, critic

def classic_model(observation_space, action_space):
    encoder_f = em.encoder_simple_dense

    actor = am.actor_simple_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_simple_dense(observation_space,action_space,encoder_f)

    return actor, critic

def unity_res_ddpg(observation_space, action_space):
    encoder_f = em.encoder_simple_res

    actor = am.actor_simple_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_simple_dense(observation_space,action_space,encoder_f)

    return actor, critic
    
def unity_res_iqn(observation_space, action_space):
    encoder_f = em.encoder_simple_res

    actor = am.actor_simple_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_dense_iqn(observation_space, action_space, encoder_f)

    return actor, critic

def unity_res_iqn_icm(observation_space, action_space):
    encoder_f = em.encoder_simple_res

    actor = am.actor_simple_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_dense_iqn(observation_space, action_space, encoder_f)

    icm_models = ICM.ICM_dense(observation_space, action_space, encoder_f)

    return actor, critic, icm_models

def unity_conv_iqn_icm(observation_space, action_space):
    encoder_f = em.encoder_simple_conv

    actor = am.actor_simple_dense(observation_space, action_space, encoder_f)

    critic = cm.critic_dense_iqn(observation_space, action_space, encoder_f)

    icm_models = ICM.ICM_dense(observation_space, action_space, encoder_f)

    return actor, critic, icm_models


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
