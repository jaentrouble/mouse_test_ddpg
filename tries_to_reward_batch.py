import gym
import gym_mouse
import argparse
from Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools
from pathlib import Path
import tqdm
import json
import ffmpeg

TEST_PER_MODEL = 10
MAX_TRIAL = 100
FRAMERATE = 10
TARGET_MODELS = [8,16,24,32,40,48,56,64]

ENVIRONMENT = 'mouseUnity-v0'
env_kwargs = dict(
    ip='localhost',
    port=7778,
)

parser = argparse.ArgumentParser()
parser.add_argument('-l','--load',dest='load',required=True)
args = parser.parse_args()

model_f = am.unity_res_model

save_dir = Path(args.load)
result_dir = save_dir/'tries_to_reward'
if not result_dir.exists():
    result_dir.mkdir()
vid_dir = result_dir/'videos'
if not vid_dir.exists():
    vid_dir.mkdir()

model_list = [save_dir/str(t) for t in TARGET_MODELS]
model_list.sort()

log_name = f'{args.load}_tries_to_reward'

env = gym.make(ENVIRONMENT, **env_kwargs)

player = Player(
    observation_space= env.observation_space, 
    action_space= env.action_space, 
    model_f= model_f,
    tqdm= None,
    m_dir=None,
    log_name= log_name,
    mixed_float=True,
)

all_try_results=[]
all_steps_results = []

for model_dir in tqdm.tqdm(
    model_list,
    unit='model',
):
    player.reload_model(str(model_dir))
    tries_results = []
    steps_results = []
    for i in tqdm.trange(
        TEST_PER_MODEL,
        unit='test',
        leave=False,
    ):
        rewarded = False
        tries = 0
        try_tqdm = tqdm.tqdm(unit='try',total=MAX_TRIAL, leave=False)
        while (not rewarded) and tries < MAX_TRIAL:
            tries += 1
            try_tqdm.update()
            speed_tqdm = tqdm.tqdm(unit='step',leave=False)
            
            done=False
            steps=0
            obs_name = f'{model_dir.name}_{i}_{tries}_obs.mp4'
            ren_name = f'{model_dir.name}_{i}_{tries}_ren.mp4'

            obs_size = env.observation_space['obs'].shape[1::-1]

            obs_out = (
                ffmpeg
                .input('pipe:',format='rawvideo',pix_fmt='rgb24',
                        s=f'{obs_size[0]}x{obs_size[1]}',
                        loglevel='panic',framerate=10)
                .output(str(vid_dir/obs_name),pix_fmt='yuv420p',
                        video_bitrate='1M')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            ren_out = (
                ffmpeg
                .input('pipe:',format='rawvideo',pix_fmt='rgb24',
                        s=f'{env.render_size[0]}x{env.render_size[1]}',
                        loglevel='panic',framerate=10)
                .output(str(vid_dir/ren_name),pix_fmt='yuv420p',
                        video_bitrate='1M')
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            o = env.reset()
            while not done:
                steps += 1
                speed_tqdm.update()
                a = player.act(o, evaluate=False)
                o,r,done,_ = env.step(a)
                obs_out.stdin.write(
                    o['obs'][...,-3:].copy(order='C')
                )
                ren = env.render('rgb')
                ren_out.stdin.write(ren.copy(order='C'))
                if r>0 :
                    rewarded = True
                    break
            obs_out.stdin.close()
            ren_out.stdin.close()
            speed_tqdm.close()
        try_tqdm.close()
        tries_results.append(tries)
        steps_results.append(steps)
    all_try_results.append(tries_results)
    all_steps_results.append(steps_results)
            
with open(result_dir/'trial_result.json','w') as f:
    json.dump(all_try_results, f)

with open(result_dir/'steps_result.json','w') as f:
    json.dump(all_steps_results, f)
