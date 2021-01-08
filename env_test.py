import gym
import gym_mouse
import time
from tqdm import trange
import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('unitytest.mp4',fourcc,60, (80,80))

env_kwargs = dict(
    ip='localhost',
    port = 7777,
)


st = time.time()
env = gym.make('mouseUnity-v0', **env_kwargs)
env.reset()
# diff = 0
# for _ in trange(100):
    # diff += env.check_step(env.action_space.sample())
for _ in trange(100):
    o, r, d, i = env.step(env.action_space.sample())
    writer.write(o['obs'][...,-3:])
    if d :
        env.reset()
    env.render()
# print(diff)
# input('done:')
writer.release()