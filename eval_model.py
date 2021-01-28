import gym
import gym_mouse
import time
from tqdm import trange
import numpy as np
import cv2
import argparse
from Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools


parser = argparse.ArgumentParser()
parser.add_argument('-l','--load', dest='load', required=True,)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)

args = parser.parse_args()


model_f = am.classic_iqn

evaluate_f = tools.evaluate_common

env_kwargs = dict(
)
ENVIRONMENT = 'LunarLanderContinuous-v2'

st = time.time()
env = tools.EnvWrapper(gym.make(ENVIRONMENT, **env_kwargs))
env.reset()
player = Player(
    observation_space=env.observation_space,
    action_space= env.action_space, 
    model_f= model_f,
    tqdm= None,
    mixed_float=args.mixed_float,
    m_dir=args.load,
)

score = evaluate_f(player, env, 'mp4', fps=30)
