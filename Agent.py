import tensorflow as tf
from tensorflow import math as tm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.clip_ops import global_norm
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
from tensorflow.keras import mixed_precision
from functools import partial

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
                 log_name=None, start_step=0, start_round=0, mixed_float=False):
        """
        Parameters
        ----------
        observation_space : gym.Space
            Observation space of the environment.
        action_space : gym.Space
            Action space of the environment. Current agent expects only
            a discrete action space.
        model_f
            A function that returns actor, critic models. 
            It should take obeservation space and action space as inputs.
            It should not compile the model.
        tqdm : tqdm.tqdm
            A tqdm object to update every step.
        m_dir : str
            A model directory to load the model if there's a model to load
        log_name : str
            A name for log. If not specified, will be set to current time.
            - If m_dir is specified yet no log_name is given, it will continue
            counting.
            - If m_dir and log_name are both specified, it will load model from
            m_dir, but will record as it is the first training.
        start_step : int
            Total step starts from start_step
        start_round : int
            Total round starts from start_round
        mixed_float : bool
            Whether or not to use mixed precision
        """
        # model : The actual training model
        # t_model : Fixed target model
        print('Model directory : {}'.format(m_dir))
        print('Log name : {}'.format(log_name))
        print('Starting from step {}'.format(start_step))
        print('Starting from round {}'.format(start_round))
        print(f'Use mixed float? {mixed_float}')
        self.tqdm = tqdm
        self.action_space = action_space
        self.action_range = action_space.high - action_space.low
        self.action_shape = action_space.shape
        self.observation_space = observation_space
        self.mixed_float = mixed_float
        if mixed_float:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)


        # Ornstein-Uhlenbeck process
        self.last_oup = 0

        # Soft Actor Critic variable
        self.soft_alpha = tf.Variable(0.0, trainable=True,dtype=tf.float32)
        self.target_entropy = tf.cast(
            -tf.reduce_prod(self.action_shape),dtype='float32')
        alpha_lr = tf.function(partial(self._lr, 'alpha'))
        self.alpha_optimizer = keras.optimizers.Adam(
            learning_rate=alpha_lr,
            epsilon=hp.lr['alpha'].epsilon,
            global_clipnorm=hp.lr['alpha'].grad_clip,
        )

        
        actor, critic = model_f(observation_space, action_space)
        self.models={
            'actor' : actor,
            'critic' : critic,
            'critic2' : keras.models.clone_model(critic),
        }
        if m_dir is None :
            # compile models
            for name, model in self.models.items():
                lr = tf.function(partial(self._lr, name))
                optimizer = keras.optimizers.Adam(
                    learning_rate=lr,
                    epsilon=hp.lr[name].epsilon,
                    global_clipnorm=hp.lr[name].grad_clip,
                )
                if self.mixed_float:
                    optimizer = mixed_precision.LossScaleOptimizer(
                        optimizer
                    )
                model.compile(optimizer=optimizer)
        else:
            # compile models
            for name, model in self.models.items():
                lr = tf.function(partial(self._lr, name))
                optimizer = keras.optimizers.Adam(
                    learning_rate=lr,
                    epsilon=hp.lr[name].epsilon,
                    global_clipnorm=hp.lr[name].grad_clip,
                )
                if self.mixed_float:
                    optimizer = mixed_precision.LossScaleOptimizer(
                        optimizer
                    )
                model.compile(optimizer=optimizer)
                model.load_weights(path.join(m_dir,name))
            print('model loaded')
        self.t_models = {}
        for name, model in self.models.items():
            self.t_models[name] = keras.models.clone_model(model)
            self.t_models[name].set_weights(model.get_weights())
            model.summary()

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
            if log_name is None :
                self.save_dir, self.save_count = path.split(m_dir)
                self.save_count = int(self.save_count)
            else:
                self.save_dir = path.join('savefiles',
                                        self.log_name)
                self.save_count = 0
        self.model_dir = None

    def _lr(self, name):
        effective_steps = self.total_steps - hp.Learn_start\
                                           - hp.lr[name].halt_steps
        if tf.greater(effective_steps, int(hp.lr[name].nsteps)):
            return hp.lr[name].end
        elif tf.less(effective_steps, 0):
            return 0.0
        else :
            new_lr = hp.lr[name].start*\
                ((hp.lr[name].end/hp.lr[name].start)**\
                    (tf.cast(effective_steps,tf.float32)/hp.lr[name].nsteps))
            return new_lr

    # @property
    # @tf.function
    # def oup_stddev(self):
    #     if tf.greater(self.total_steps, hp.OUP_stddev_nstep) :
    #         return hp.OUP_stddev_end
    #     else:
    #         return tf.cast(hp.OUP_stddev_start+\
    #             (hp.OUP_stddev_end-hp.OUP_stddev_start)*\
    #             (self.total_steps/hp.OUP_stddev_nstep),dtype=tf.float32)

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
        action, _ = self.models['actor'](processed_state, training=False)
        # if self.total_steps % hp.log_per_steps==0:
        #     tf.summary.scalar('a0_raw', action[0][0], self.total_steps)
        #     tf.summary.scalar('a1_raw', action[0][1], self.total_steps)
        return action

    # @tf.function
    # def choose_action_no_noise(self, before_state):
    #     """
    #     Policy part
    #     For evaluation; no noise is added
    #     """
    #     processed_state = self.pre_processing(before_state)
    #     raw_action, _ = self.models['actor'](processed_state, training=False)
    #     action = raw_action
    #     return action

    


    def act_batch(self, before_state, evaluate=False):
        # if evaluate:
        #     action = self.choose_action_no_noise(before_state)
        # else:
        action = self.choose_action(before_state)
        return action.numpy()
        
    def act(self, before_state, evaluate=False):
        """
        Will squeeze axis=0 if Batch_num = 1
        If you don't want to squeeze, use act_batch()
        
        If eval = True, noise is not added
        """
        # if evaluate:
        #     action = self.choose_action_no_noise(before_state)
        # else:
        action = self.choose_action(before_state)
        action_np = action.numpy()
        if action_np.shape[0] == 1:
            if self.total_steps % hp.log_per_steps==0 and not evaluate:
                tf.summary.scalar('a0', action[0][0], self.total_steps)
            return action_np[0]
        else:
            if self.total_steps % hp.log_per_steps==0 and not evaluate:
                tf.summary.scalar('a0', action[0], self.total_steps)
            return action_np


    # @tf.function
    # def oup_noise(self, action):
    #     """
    #     Add Ornstein-Uhlenbeck noise to action
    #     """
    #     noise = (1 - hp.OUP_damping)*self.last_oup + \
    #             tf.random.normal(
    #                 shape=self.action_shape, 
    #                 mean=0.0,
    #                 stddev=self.oup_stddev,
    #             )*self.action_range
    #     self.last_oup = noise
    #     return action + noise

    @tf.function
    def train_step(self, o, r, d, a, sp_batch, weights):
        """
        All inputs are expected to be preprocessed
        """
        # next Q values from t_critic to evaluate
        # In Soft Actor Critic, current policy is used
        next_action, log_pi = self.models['actor'](sp_batch, training=False)

        # add action, tau to input
        t_critic_input = sp_batch.copy()
        t_critic_input['action'] = next_action
        target_q1 = self.t_models['critic'](
            t_critic_input, 
            training=False,
        )
        target_q2 = self.t_models['critic2'](
            t_critic_input, 
            training=False,
        )
        target_q_min = tf.minimum(target_q1, target_q2)
        soft_target = target_q_min - self.soft_alpha * log_pi
        critic_target = r + \
                        tf.cast(tm.logical_not(d),tf.float32)*\
                        hp.Q_discount * \
                        soft_target

        # First update critic1 & 2
        with tf.GradientTape() as critic1_tape:
            # add action to input
            critic1_input = o.copy()
            critic1_input['action'] = a
            q1 = self.models['critic'](
                critic1_input,
                training=True,
            )
            critic1_unweighted_loss = tf.reduce_mean((critic_target-q1)**2)
            critic1_loss = tf.math.reduce_mean(weights * critic1_unweighted_loss)
            critic1_loss_original = critic1_loss
            if self.mixed_float:
                critic1_loss = self.models['critic'].optimizer.get_scaled_loss(
                    critic1_loss
                )
        with tf.GradientTape() as critic2_tape:
            # add action to input
            critic2_input = o.copy()
            critic2_input['action'] = a
            q2 = self.models['critic2'](
                critic2_input,
                training=True,
            )
            critic2_unweighted_loss = tf.reduce_mean((critic_target-q2)**2)
            critic2_loss = tf.math.reduce_mean(weights * critic2_unweighted_loss)
            critic2_loss_original = critic2_loss
            if self.mixed_float:
                critic2_loss = self.models['critic'].optimizer.get_scaled_loss(
                    critic2_loss
                )

        if self.total_steps % hp.log_per_steps==0:
            tf.summary.scalar('Critic Loss', critic1_loss_original, self.total_steps)
            tf.summary.scalar('q', tf.math.reduce_mean(q1), self.total_steps)
            tf.summary.scalar('log_pi',tf.math.reduce_mean(log_pi),self.total_steps)
            

        critic1_vars = self.models['critic'].trainable_weights
        critic1_gradients = critic1_tape.gradient(critic1_loss, critic1_vars)
        if self.mixed_float:
            critic1_gradients = \
                self.models['critic'].optimizer.get_unscaled_gradients(
                    critic1_gradients
                )
        self.models['critic'].optimizer.apply_gradients(
            zip(critic1_gradients, critic1_vars)
        )

        critic2_vars = self.models['critic2'].trainable_weights
        critic2_gradients = critic2_tape.gradient(critic2_loss, critic2_vars)
        if self.mixed_float:
            critic2_gradients = \
                self.models['critic2'].optimizer.get_unscaled_gradients(
                    critic2_gradients
                )
        self.models['critic2'].optimizer.apply_gradients(
            zip(critic2_gradients, critic2_vars)
        )


        # Then update actor
        with tf.GradientTape() as actor_tape:
            action, log_pi = self.models['actor'](o, training=True)
            # New input
            critic_input = o.copy()
            critic_input['action'] = action
            # Shape (batch, support)
            q1 = self.models['critic'](
                critic_input,
                training=False,
            )
            q2 = self.models['critic2'](
                critic_input,
                training=False,
            )
            q_min = tf.minimum(q1, q2)
            soft_q = q_min - self.soft_alpha * log_pi
            # Actor needs to 'ascend' gradient
            J = (-1.0) * tf.reduce_mean(soft_q)
            if self.mixed_float:
                J = self.models['actor'].optimizer.get_scaled_loss(J)

        actor_vars = self.models['actor'].trainable_weights

        actor_gradients = actor_tape.gradient(J, actor_vars)
        if self.mixed_float:
            actor_gradients = \
                self.models['actor'].optimizer.get_unscaled_gradients(
                    actor_gradients
                )

        self.models['actor'].optimizer.apply_gradients(
            zip(actor_gradients, actor_vars)
        )

        # Finally, update Soft Actor Critic alpha
        action, log_pi = self.models['actor'](o, training=False)
        with tf.GradientTape() as alpha_tape:
            alpha_loss = tf.reduce_mean(
                -self.soft_alpha * (log_pi + self.target_entropy)
            )
        alpha_vars = [self.soft_alpha]
        alpha_gradients = alpha_tape.gradient(alpha_loss, alpha_vars)
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients,alpha_vars))

        if self.total_steps % hp.log_per_steps==0:
            tf.summary.scalar(
                'soft_alpha',
                self.soft_alpha,
                step=self.total_steps,
            )

        tf.summary.scalar(
            'critic_grad_norm',
            tf.linalg.global_norm(critic1_gradients),
            step=self.total_steps,
        )
        tf.summary.scalar(
            'actor_grad_norm',
            tf.linalg.global_norm(actor_gradients),
            step=self.total_steps,
        )

        priority = ((critic1_unweighted_loss+critic2_unweighted_loss)/2\
                    +hp.Buf.epsilon)**hp.Buf.alpha
        return priority


    def step(self, before, action, reward, done, info):
        self.buffer.store_step(before, action, reward, done)
        self.tqdm.update()
        # Record here, so that it won't record when evaluating
        self.cumreward += reward
        if self.total_steps % hp.log_per_steps==0:
            for name in self.models:
                tf.summary.scalar(f'lr_{name}',self._lr(name),self.total_steps)
        if done:
            tf.summary.scalar('Reward', self.cumreward, self.rounds)
            tf.summary.scalar('Reward_step', self.cumreward, self.total_steps)
            tf.summary.scalar('Steps_per_round',self.current_steps,self.rounds)
            info_dict = {
                'Round':self.rounds,
                'Steps':self.current_steps,
                'Reward':self.cumreward,
            }
            self.tqdm.set_postfix(info_dict)
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
        print('saving model..')
        self.save_count += 1
        self.model_dir = path.join(self.save_dir, str(self.save_count))
        if not path.exists(self.model_dir):
            makedirs(self.model_dir)
        for name, model in self.models.items():
            weight_dir = path.join(self.model_dir,name)
            model.save_weights(weight_dir)
        # print('saving buffer..')
        # with open(path.join(self.model_dir,'buffer.bin'),'wb') as f :
        #     pickle.dump(self.buffer, f)

        return self.save_count

