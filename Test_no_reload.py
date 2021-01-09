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
from tensorflow.keras import mixed_precision
from datetime import timedelta

ENVIRONMENT = 'mouseUnity-v0'

env_kwargs = dict(
    ip='localhost',
    port = 7777,
)

model_f = am.unity_res_model

evaluate_f = tools.evaluate_unity

parser = argparse.ArgumentParser()
parser.add_argument('-r','--render', dest='render',action='store_true', default=False)
parser.add_argument('--step', dest='total_steps',default=100000)
parser.add_argument('-n','--logname', dest='log_name',default=False)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
parser.add_argument('-lr', dest='lr', default=1e-5, type=float)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
args = parser.parse_args()

if args.mixed_float:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


vid_type = 'mp4'
total_steps = int(args.total_steps)
my_tqdm = tqdm(total=total_steps, dynamic_ncols=True)


hp.Model_save = 2000
hp.Learn_start = 100

hp.lr_start = args.lr
hp.lr_end = hp.lr_start * 1e-5
hp.lr_nsteps = 1000000


if args.render :
    from gym.envs.classic_control.rendering import SimpleImageViewer
    eye_viewer = SimpleImageViewer(maxwidth=1500)
# For benchmark
st = time.time()

need_to_eval = False

env = gym.make(ENVIRONMENT, **env_kwargs)
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
        if ((hp.Learn_start + 25 + step) % hp.Model_save) == 0 :
            need_to_eval = True
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            if need_to_eval:
                player.save_model()
                score = evaluate_f(player, env, vid_type)
                print('eval_score:{0}'.format(score))
                need_to_eval = False

            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

else :
    for step in range(total_steps):
        if (step>0) and ((step % hp.Model_save) == 0) :
            need_to_eval = True
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            if need_to_eval:
                player.save_model()
                score = evaluate_f(player, env, vid_type)
                print('eval_score:{0}'.format(score))
                need_to_eval = False

            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

player.save_model()
score = evaluate_f(player, env, vid_type)
print('eval_score:{0}'.format(score))
d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')
my_tqdm.close()

