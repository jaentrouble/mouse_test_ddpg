import gym
from gym.spaces import Discrete, Dict, Box
from gym.utils import seeding
import numpy as np
import socket
import json
import sys

# NOTE: (WIDTH, HEIGHT)
FRAME_SIZE = (80,80)
FRAME_BYTE = np.prod(FRAME_SIZE) * 4 #Channel 4
RENDER_SIZE = (192,192)
RENDER_BYTE = np.prod(RENDER_SIZE) * 4 #Channel 4


RECV_BYTE = 16384

MAX_SPEED = 0.5
MAX_ANGLE = 10

NUTELLA_REWARD = 100
PUNISH_STEP = 1
PUNISH_DIST = 0

FINISH_WHEN_REWARD = True

class MouseEnv_unity(gym.Env) :
    """MouseEnv_unity

    """
    metadata = {
        'render.modes' : ['human','rgb']
    }
    def __init__(self, **kwargs):
        """
        kwargs
        ------
        ip : str
            ip adress to listen, default localhost
        port : 
            port number to listen, default 7777
        """
        self.render_size = RENDER_SIZE
        # Spin first and move
        self.action_space = Box(
            low=np.array([0.0,-1.0]),
            high=np.array([1.0,1.0]),
            dtype=np.float32
        )
        self._done = False
        self._initialized = False

        kwargs.setdefault('ip','localhost')
        kwargs.setdefault('port',7777)
        self._ip = kwargs['ip']
        self._port = kwargs['port']

        self._options = kwargs

        self.viewer = None

        self.max_step = 1000
        self.cur_step = 0
        self.seed()

        # 3 Continuous Inputs from both eyes
        self.observation_space = Dict(
            {'obs' : Box(0, 255, shape=(FRAME_SIZE[1],FRAME_SIZE[0],9), 
                                        dtype=np.uint8)}
        )
        

    def step(self, action):
        assert self._initialized, 'Reset first before starting env'
        if self._done :
            print('The game is already done. Continuing may cause unexpected'\
                ' behaviors')

        action = np.clip(action,self.action_space.low, self.action_space.high)

        to_send = {
            'move':float(action[0])*MAX_SPEED,
            'turn':float(action[1])*MAX_ANGLE,
            'reset':False,
        }
        data = self._send_and_receive(to_send)

        raw_image = np.frombuffer(data[:FRAME_BYTE],dtype=np.uint8)
        # unity renders from bottom to top
        new_obs = raw_image.reshape(
            (FRAME_SIZE[1],FRAME_SIZE[0],4)
        )[::-1,...,:3]
        self._obs_buffer.pop(0)
        self._obs_buffer.append(new_obs)
        observation = {
            'obs':np.concatenate(self._obs_buffer,axis=-1)
        }
        info_str = data[FRAME_BYTE:].decode('utf-8')
        info = json.loads(info_str)

        reward = info['reward'] * NUTELLA_REWARD
        done = info['done'] or (reward>0)

        # Movement punishment
        dist = action[0]*MAX_SPEED
        theta = action[1]*MAX_ANGLE*np.pi/180
        x = np.sqrt(np.absolute(
            1 + (1+dist)**2 - 2*(1+dist)*np.cos(theta)
        ))
        p = PUNISH_STEP + x*PUNISH_DIST
        reward -= p


        if done:
            self._done = True
        #Check if reached max_step
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            self._done = True
            done = True

        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment and return initial observation
        """
        if not self._initialized:
            self._initialized = True
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self._ip, self._port))
                s.listen()
                print('listening')
                self.conn, addr = s.accept()
                print('connected by',addr)
        
        self._done = False
        self.cur_step = 0

        to_send = {
            'move' : 0.0,
            'turn' : 0.0,
            'reset' : True,
        }
        data = self._send_and_receive(to_send)

        raw_image = np.frombuffer(data[:25600],dtype=np.uint8)
        # unity renders from bottom to top
        new_obs = raw_image.reshape(
            (FRAME_SIZE[1],FRAME_SIZE[0],4)
        )[::-1,...,:3]

        self._obs_buffer = [new_obs]*3
        initial_observation = {
            'obs':np.concatenate(self._obs_buffer,axis=-1)
        }
        return initial_observation

    def render(self, mode='human'):
        assert self._initialized, 'Reset first before starting env'
        to_send = {
            'render' : True,
        }
        data = self._send_and_receive(to_send)
        image_raw = np.frombuffer(data,dtype=np.uint8)
        # unity renders from bottom to top
        image = image_raw.reshape(
            (RENDER_SIZE[1],RENDER_SIZE[0],4)
        )[::-1,...,:3]
        if 'human' in mode :
            from gym.envs.classic_control import rendering
            if self.viewer == None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=720)
            self.viewer.imshow(image)
        elif 'rgb' in mode :
            return image

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _send_and_receive(self, to_send:dict):
        """
        send JSON to client and return received raw data in bytes.
        """
        to_send_byte = json.dumps(to_send).encode('utf-8')
        self.conn.sendall(to_send_byte)
        recv_info = self.conn.recv(RECV_BYTE)
        if not recv_info:
            raise ConnectionAbortedError
        recv_size = int(recv_info.decode('utf-8'))
        self.conn.sendall(recv_info)
        
        all_data = []
        received_bytes = 0
        while received_bytes<recv_size:
            data = self.conn.recv(RECV_BYTE)
            if not data:
                raise ConnectionAbortedError
            all_data.append(data)
            received_bytes += len(data)
        return b''.join(all_data)


# Testing
if __name__ == '__main__' :
    env = MouseEnv_cl()
    env.render()
    a = input()