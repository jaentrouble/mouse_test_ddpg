import gym
import gym_mouse
import time
import numpy as np
from Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools
import agent_assets.A_hparameters as hp
from tqdm import tqdm
import argparse
import os
import sys
from tensorflow.profiler.experimental import Profile
from datetime import timedelta
from sanity_env import EnvTest

ENVIRONMENT = 'mouseClCont-v0'

env_kwargs = dict(
    apple_num=10,
    eat_apple = 1.0,
    hit_wall = 0,
)

hp.Buffer_size = 10000
hp.Learn_start = 2000
hp.Batch_size = 32
hp.Target_update = 10
hp.Target_update_tau = 1e-2

hp.lr_start=1e-5
hp.lr_end = 1e-10

hp.OUP_stddev = 0.2
hp.OUP_stddev_min = 0.05
hp.OUP_stddev_nstep = 50000

model_f = am.eye_brain_model


parser = argparse.ArgumentParser()
parser.add_argument('-r','--render', dest='render',action='store_true', default=False)
parser.add_argument('--step', dest='total_steps',default=100000)
parser.add_argument('-n','--logname', dest='log_name',default=False)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
args = parser.parse_args()

vid_type = 'mp4'
total_steps = int(args.total_steps)
my_tqdm = tqdm(total=total_steps, dynamic_ncols=True)

if args.render :
    from gym.envs.classic_control.rendering import SimpleImageViewer
    eye_viewer = SimpleImageViewer(maxwidth=1500)
    bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
# For benchmark
st = time.time()

original_env = gym.make(ENVIRONMENT, **env_kwargs)
env = EnvTest(original_env.observation_space)

bef_o = env.reset()

if args.log_name:
    # If log directory is explicitely selected
    player = Player(
        observation_space= env.observation_space, 
        action_space= env.action_space, 
        model_f= model_f,
        tqdm= my_tqdm,
        log_name= args.log_name
    )
else :
    player = Player(
        observation_space= env.observation_space,
        action_space= env.action_space, 
        model_f= model_f,
        tqdm= my_tqdm,
    )
if args.render :
    env.render()

if args.profile:
    # Warm up
    for step in range(hp.Learn_start+20):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

    with Profile(f'logs/{args.log_name}'):
        for step in range(5):
            action = player.act(bef_o)
            aft_o,r,d,i = env.step(action)
            player.step(bef_o,action,r,d,i)
            if d :
                bef_o = env.reset()
            else:
                bef_o = aft_o
            if args.render :
                env.render()
    remaining_steps = total_steps - hp.Learn_start - 25
    for step in range(remaining_steps):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

else :
    for step in range(total_steps):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')
my_tqdm.close()

