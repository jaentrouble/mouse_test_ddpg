import gym
import argparse
from Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools

parser = argparse.ArgumentParser()
parser.add_argument('-l','--load',dest='load',required=True)
args = parser.parse_args()

model_f = am.unity_res_model

