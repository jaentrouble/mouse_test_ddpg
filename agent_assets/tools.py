import cv2
from os import path, makedirs
import numpy as np

def evaluate_unity(player, env, video_type):
    print('Evaluating...')
    done = False
    if player.model_dir is None:
        eval_dir = path.join(player.save_dir,'eval')
        if not path.exists(eval_dir):
            makedirs(eval_dir)
    else:
        eval_dir = player.model_dir
    eye_dir = path.join(eval_dir,f'eval_eye.{video_type}')
    ren_dir = path.join(eval_dir,f'eval_ren.{video_type}')
    score_dir = path.join(eval_dir,'score.txt')

    if 'avi' in video_type :
        fcc = 'DIVX'
    elif 'mp4' in video_type:
        fcc = 'mp4v'
    else:
        raise TypeError('Wrong videotype')
    fourcc = cv2.VideoWriter_fourcc(*fcc)
    # Becareful : cv2 order of image size is (width, height)
    eye_size = env.observation_space['obs'].shape[1::-1]
    eye_out = cv2.VideoWriter(eye_dir, fourcc, 10, eye_size)
    ren_out = cv2.VideoWriter(ren_dir, fourcc, 10, env.render_size)

    o = env.reset()
    score = 0
    loop = 0
    while not done :
        loop += 1
        if not loop % 100:
            print('Eval : {}step passed'.format(loop))
        a = player.act(o, evaluate=True)
        o,r,done,i = env.step(a)
        score += r
        #eye recording
        eye_out.write(o['obs'][...,-1:-4:-1])
        ren_out.write(env.render('rgb')[...,::-1])
    ren_out.release()
    eye_out.release()
    with open(score_dir, 'w') as f:
        f.write(str(score))
    print('Eval finished')
    return score

def evaluate_common(player, env, video_type):
    print('Evaluating...')
    done = False
    video_dir = path.join(player.model_dir, 'eval.{}'.format(video_type))
    score_dir = path.join(player.model_dir, 'score.txt')
    if 'avi' in video_type :
        fcc = 'DIVX'
    elif 'mp4' in video_type:
        fcc = 'mp4v'
    else:
        raise TypeError('Wrong videotype')
    fourcc = cv2.VideoWriter_fourcc(*fcc)
    # Becareful : cv2 order of image size is (width, height)
    o = env.reset()
    rend_img = env.render('rgb_array')
    # cv2 expects 90 degrees rotated
    out_shape = (rend_img.shape[1],rend_img.shape[0])
    out = cv2.VideoWriter(video_dir, fourcc, 10, out_shape)
    score = 0
    loop = 0
    while not done :
        loop += 1
        if loop % 100 == 0:
            print('Eval : {}step passed'.format(loop))
        a = player.act(o, evaluate=True)
        o,r,done,i = env.step(a)
        score += r
        # This will turn image 90 degrees, but it does not make any difference,
        # so keep it this way to save computations
        img = env.render('rgb_array')
        out.write(np.flip(env.render('rgb_array'), axis=-1))
    out.release()
    with open(score_dir, 'w') as f:
        f.write(str(score))
    print('Eval finished')
    env.close()
    return score

class EnvWrapper():
    """Change normal observation into dictionary type
    {
        'obs' : (actual observation)
    }
    """
    def __init__(self, env):
        self.env = env
        from gym import spaces
        self.observation_space = spaces.Dict(
            {'obs' : self.env.observation_space}
        )
    
    def __getattr__(self, attr):
        return self.env.__getattribute__(attr)

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return {'obs':o}, r, d, i

    def reset(self):
        return {'obs':self.env.reset()}