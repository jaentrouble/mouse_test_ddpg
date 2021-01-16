import gym
import gym_mouse
import time
from tqdm import trange
import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('unitytest.mp4',fourcc,60, (80,80))
writer_ren = cv2.VideoWriter('unitytest_ren.mp4',fourcc,60, (192,192))

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
for _ in trange(3000):
    o, r, d, i = env.step(np.array([0.3,0.3]))
    writer.write(o['obs'][...,-1:-4:-1])
    writer_ren.write(env.render('rgb')[...,::-1])
    if d :
        env.reset()
    
    if r > 0:
        print(r)
# print(diff)
# input('done:')
writer.release()
writer_ren.release()