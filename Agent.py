import tensorflow as tf
from tensorflow import math as tm
from tensorflow import keras
from tensorflow.keras import layers
import agent_assets.A_hparameters as hp
from datetime import datetime
from os import path, makedirs
import random
import cv2
import numpy as np
from agent_assets.replaybuffer import ReplayBuffer
from agent_assets.mousemodel import QModel
import pickle
from tqdm import tqdm

#leave memory space for opencl
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

keras.backend.clear_session()

class Player():
    """A agent class which plays the game and learn.

    Algorithms
    ----------
    DDPG
    Prioritized sampling
    """
    def __init__(self, observation_space, action_space, model_f, tqdm, m_dir=None,
                 log_name=None, start_step=0, start_round=0,load_buffer=False):
        """
        Parameters
        ----------
        observation_space : gym.Space
            Observation space of the environment.
        action_space : gym.Space
            Action space of the environment. Current agent expects only
            a discrete action space.
        model_f
            A function that returns encoder, actor, critic models. 
            It should take obeservation space and action space as inputs.
            It should not compile the model.
        tqdm : tqdm.tqdm
            A tqdm object to update every step.
        m_dir : str
            A model directory to load the model if there's a model to load
        log_name : str
            A name for log
        start_step : int
            Total step starts from start_step
        start_round : int
            Total round starts from start_round
        load_buffer : bool
            Whether to load the buffer from the model directory
        """
        # model : The actual training model
        # t_model : Fixed target model
        print('Model directory : {}'.format(m_dir))
        print('Log name : {}'.format(log_name))
        print('Starting from step {}'.format(start_step))
        print('Starting from round {}'.format(start_round))
        print('Load buffer? {}'.format(load_buffer))
        self.tqdm = tqdm
        self.action_space = action_space
        self.action_range = action_space.high - action_space.low
        self.action_shape = action_space.shape
        self.observation_space = observation_space

        # Ornstein-Uhlenbeck process
        self.last_oup = 0

        #Inputs
        if m_dir is None :
            encoder, actor, critic = model_f(observation_space, action_space)
            self.models={
                'encoder': encoder,
                'actor' : actor,
                'critic' : critic,
            }
            # compile models
            optimizer = keras.optimizers.Adam(learning_rate=self._lr)
            for model in self.models.values():
                model.compile(optimizer=optimizer)
        else:
            encoder, actor, critic = model_f(observation_space, action_space)
            self.models={
                'encoder': encoder,
                'actor' : actor,
                'critic' : critic,
            }
            # compile models
            optimizer = keras.optimizers.Adam(learning_rate=self._lr)
            for name, model in self.models.items():
                model.compile(optimizer=optimizer)
                model.load_weights(path.join(m_dir,name))
            print('model loaded')
        self.t_models = {}
        for name, model in self.models.items():
            self.t_models[name] = keras.models.clone_model(model)
            self.t_models[name].set_weights(model.get_weights())
            model.summary()

        # Buffers
        if load_buffer:
            print('loading buffers...')
            with open(path.join(m_dir,'buffer.bin'),'rb') as f :
                self.buffer = pickle.load(f)
            print('loaded : {} filled in buffer'.format(self.buffer.num_in_buffer))
            print('Current buffer index : {}'.format(self.buffer.next_idx))
        else :
            self.buffer = ReplayBuffer(hp.Buffer_size, self.observation_space,
                                       self.action_space)

        # File writer for tensorboard
        if log_name is None :
            self.log_name = datetime.now().strftime('%m_%d_%H_%M_%S')
        else:
            self.log_name = log_name
        self.file_writer = tf.summary.create_file_writer(path.join('logs',
                                                         self.log_name))
        self.file_writer.set_as_default()
        print('Writing logs at logs/'+ self.log_name)

        # Scalars
        self.start_training = False
        self.total_steps = tf.Variable(start_step, dtype=tf.int64)
        self.current_steps = 1
        # self.score = 0
        self.rounds = start_round
        self.cumreward = 0
        
        # Savefile folder directory
        if m_dir is None :
            self.save_dir = path.join('savefiles',
                            self.log_name)
            self.save_count = 0
        else:
            self.save_dir, self.save_count = path.split(m_dir)
            self.save_count = int(self.save_count)

    @tf.function
    def _lr(self):
        new_lr = hp.lr_start*\
            ((hp.lr_end/hp.lr_start)**\
                (tf.cast(self.total_steps,tf.float32)/hp.lr_nsteps))
        if tf.less(new_lr, 1e-35):
            return 1e-35
        else :
            return new_lr

    @property
    @tf.function
    def epsilon(self):
        if tf.greater(self.total_steps, hp.epsilon_nstep) :
            return hp.epsilon_min
        else:
            return tf.cast(hp.epsilon-(hp.epsilon-hp.epsilon_min)*\
                (self.total_steps/hp.epsilon_nstep),dtype=tf.float32)

    @tf.function
    def pre_processing(self, observation:dict):
        """
        Preprocess input data
        """
        processed_obs = {}
        for name, obs in observation.items():
            # If only one observation is given, reshape to [1,...]
            if len(observation[name].shape)==\
                len(self.observation_space[name].shape):
                processed_obs[name] = tf.cast(obs[tf.newaxis,...],tf.float32)/255
            else :
                processed_obs[name] = tf.cast(obs, tf.float32)/255
        return processed_obs

    @tf.function
    def choose_action(self, before_state):
        """
        Policy part
        """
        processed_state = self.pre_processing(before_state)
        encoded_state = self.models['encoder'](processed_state)
        raw_action = self.models['actor'](encoded_state)
        action = self.oup_noise(raw_action)
        return action


    def act_batch(self, before_state, record=True):
        action = self.choose_action(before_state)
        if record:
            pass
        return action.numpy()
        
    def act(self, before_state, record=True):
        """
        Will squeeze axis=0 if Batch_num = 1
        If you don't want to squeeze, use act_batch()
        """
        action = self.choose_action(before_state)
        if record:
            pass
        action_np = action.numpy()
        if action_np.shape[0] == 1:
            return action_np[0]
        else:
            return action_np


    @tf.function
    def oup_noise(self, action):
        """
        Add Ornstein-Uhlenbeck noise to action
        """
        noise = (1 - hp.OUP_damping)*self.last_oup + \
                tf.random.normal(
                    shape=self.action_shape, 
                    mean=0.0,
                    stddev=hp.OUP_stddev,
                )*self.action_range
        self.last_oup = noise
        
        return action + noise

    @tf.function
    def train_step(self, o, r, d, a, sp_batch, weights):
        """
        All inputs are expected to be preprocessed
        """

        t_encoded_sp_batch = self.t_models['encoder'](sp_batch, training=False)
        # next Q values from t_critic to evaluate
        t_action = self.t_models['actor'](t_encoded_sp_batch, training=False)
        target_q = self.t_models['critic'](
            [t_action, t_encoded_sp_batch], 
            training=False,
        )

        critic_target = r + tf.cast(tm.logical_not(d), tf.float32) * \
                            hp.Q_discount * \
                            target_q

        # First update critic
        with tf.GradientTape() as critic_tape:
            encoded_o = self.models['encoder'](o, training=True)
            with critic_tape.stop_recording():
                action = self.models['actor'](encoded_o, training=False)
            q = self.models['critic'](
                [action, encoded_o],
                training=True,
            )
            critic_unweighted_loss = tf.math.square(q - critic_target)
            critic_loss = tf.math.reduce_mean(weights * critic_unweighted_loss)
            if self.total_steps % hp.log_per_steps==0:
                tf.summary.scalar('Critic Loss', critic_loss, self.total_steps)

        encoder_vars = self.models['encoder'].trainable_weights
        critic_vars = self.models['critic'].trainable_weights
        critic_appended = encoder_vars + critic_vars

        critic_gradients = critic_tape.gradient(critic_loss, critic_appended)

        self.models['encoder'].optimizer.apply_gradients(
            zip(critic_gradients[:len(encoder_vars)], encoder_vars)
        )
        self.models['critic'].optimizer.apply_gradients(
            zip(critic_gradients[len(encoder_vars):], critic_vars)
        )

        # Then update actor
        with tf.GradientTape() as actor_tape:
            encoded_o = self.models['encoder'](o, training=True)
            action = self.models['actor'](encoded_o, training=True)
            q = self.models['critic'](
                [action, encoded_o],
                training=False,
            )
            # Actor needs to 'ascend' gradient
            J = (-1.0) * tf.reduce_mean(q)

        encoder_vars = self.models['encoder'].trainable_weights
        actor_vars = self.models['actor'].trainable_weights
        actor_appended = encoder_vars + actor_vars

        actor_gradients = actor_tape.gradient(J, actor_appended)

        self.models['encoder'].optimizer.apply_gradients(
            zip(actor_gradients[:len(encoder_vars)], encoder_vars)
        )
        self.models['actor'].optimizer.apply_gradients(
            zip(actor_gradients[len(encoder_vars):], actor_vars)
        )

        priority = (tf.math.abs(q-critic_target)+hp.Buf.epsilon)**hp.Buf.alpha
        return priority


    def step(self, before, action, reward, done, info):
        self.buffer.store_step(before, action, reward, done)
        self.tqdm.update()
        # Record here, so that it won't record when evaluating
        # if info['ate_apple']:
        #     self.score += 1
        self.cumreward += reward
        if self.total_steps % hp.log_per_steps==0:
            tf.summary.scalar('lr', self._lr(),self.total_steps)
        if done:
            # tf.summary.scalar('Score', self.score, self.rounds)
            tf.summary.scalar('Reward', self.cumreward, self.rounds)
            # tf.summary.scalar('Score_step', self.score, self.total_steps)
            tf.summary.scalar('Reward_step', self.cumreward, self.total_steps)
            info_dict = {
                'Round':self.rounds,
                'Steps':self.current_steps,
                # 'Score':self.score,
                'Reward':self.cumreward,
            }
            self.tqdm.set_postfix(info_dict)
            # self.score = 0
            self.current_steps = 0
            self.cumreward = 0
            self.rounds += 1

        if self.total_steps % hp.histogram == 0:
            for model in self.models.values():
                for var in model.trainable_weights:
                    tf.summary.histogram(var.name, var, step=self.total_steps)

        if self.buffer.num_in_buffer < hp.Learn_start :
            self.tqdm.set_description(
                f'filling buffer'
                f'{self.buffer.num_in_buffer}/{hp.Learn_start}'
            )

        else :
            if self.start_training == False:
                self.tqdm.set_description()
                self.start_training = True
            s_batch, a_batch, r_batch, d_batch, sp_batch, indices, weights = \
                                    self.buffer.sample(hp.Batch_size)
            s_batch = self.pre_processing(s_batch)
            sp_batch = self.pre_processing(sp_batch)
            # tf_total_steps = tf.constant(self.total_steps, dtype=tf.int64)
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)

            data = (
                s_batch,
                r_batch, 
                d_batch, 
                a_batch, 
                sp_batch, 
                # tf_total_steps,
                weights,
            )

            new_priors = self.train_step(*data).numpy()
            self.buffer.update_prior_batch(indices, new_priors)

            # Soft target update
            if self.total_steps % hp.Target_update == 0:
                for model, t_model in zip(
                    self.models.values(),self.t_models.values()
                ):
                    model_w = model.get_weights()
                    t_model_w = t_model.get_weights()
                    new_w = []
                    for mw, tw in zip(model_w, t_model_w):
                        nw = hp.Target_update_tau * mw + \
                             (1-hp.Target_update_tau) * tw
                        new_w.append(nw)
                    t_model.set_weights(new_w)

        self.total_steps.assign_add(1)
        self.current_steps += 1

    def save_model(self):
        """
        Saves the model and return next save file number
        """
        self.save_count += 1
        self.model_dir = path.join(self.save_dir, str(self.save_count))
        if not path.exists(self.model_dir):
            makedirs(self.model_dir)
        for name, model in self.models.items():
            weight_dir = path.join(self.model_dir,name)
            model.save_weights(weight_dir)
        with open(path.join(self.model_dir,'buffer.bin'),'wb') as f :
            pickle.dump(self.buffer, f)

        return self.save_count

