#!/usr/bin/env python
import keras
import tensorflow as tf
import numpy as npy
import gym
import sys
from copy import copy, deepcopy
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np
from math import fmod
import time
import os
import cv2
import datetime
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.core import Reshape
from keras.optimizers import Adam
from collections import deque

# Selecting the gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.environment_name = environment_name

        if(environment_name == 'CartPole-v0'):
            self.learning_rate = 0.0001
            state_size = 4
            action_size = 2
            hidden_size = 32

            self.model = Sequential()

            self.model.add(Dense(hidden_size, activation='linear',
                                 input_dim=state_size))
            self.model.add(Dense(action_size, activation='linear'))

            self.optimizer = Adam(lr=self.learning_rate)
            self.model.compile(loss='mse', optimizer=self.optimizer)

        elif(environment_name == 'MountainCar-v0'):
            self.learning_rate = 0.005
            state_size = 2
            action_size = 3
            hidden_size = 32

            self.model = Sequential()

            self.model.add(Dense(hidden_size, activation='linear',input_dim=state_size))
            self.model.add(Dense(action_size, activation='linear'))

            self.optimizer = Adam(lr=self.learning_rate)
            self.model.compile(loss='mse', optimizer=self.optimizer)


    def save_model_weights(self, count):
        # Helper function to save your model / weights.
        if not os.path.exists(os.path.join(self.environment_name, 'lnQ_weights')):
            os.makedirs(os.path.join(self.environment_name, 'lnQ_weights'))
        time_now = self.environment_name + str(count) + '.h5'
        time_now.replace(" ","")

        file_name = os.path.join(self.environment_name, 'lnQ_weights', time_now)
        self.model.save_weights(file_name)

        time_now = self.environment_name + str(datetime.datetime.now()) + '.h5'
        time_now.replace(" ","")
        file_name = os.path.join(self.environment_name, 'lnQ_weights', time_now)
        self.model.save_weights(file_name)
        
    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights.
        self.model.load_weights(weight_file)
        

class Replay_Memory():

    def __init__(self, environment_name, memory_size=10000, burn_in=500):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.env = gym.make(environment_name)
        self.memory_size = memory_size
        self.memory = []

        i = 0
        #Burn in modeled as number of transitions instead of number of episodes.
        while(i<burn_in):
            done = False
            s = self.env.reset() #Initialize in a random state
            while(done == False):
                action = self.env.action_space.sample() #Sample an action
                newState,reward,done,_ = self.env.step(action)
                s_next = deepcopy(newState)
    
                transition = [s,action,reward,s_next,done]
                self.append(transition)
                s = deepcopy(s_next)
                i +=1

        print("Memory burnt-in")


    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.

        self.memory_batch = []

        rand_num = np.random.randint(0,len(self.memory)-1,size=batch_size)

        for k in range(batch_size):
            self.memory_batch.append(self.memory[rand_num[k]])

        return self.memory_batch

    def append(self, transition):
        # Appends transition to the memory.
        if(len(self.memory)<=self.memory_size):
            self.memory.append(transition)
        else:
            del self.memory[0]
            self.memory.append(transition)


class lnQ_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.mainQN = QNetwork(environment_name)
        # self.targetQN = QNetwork(environment_name)
        self.env = gym.make(environment_name)
        time_now = str(datetime.datetime.now())
        time_now.replace(" ","")

        str_c = './tmp/'+ environment_name + '/lnQ-experiment/' + time_now
        self.env = wrappers.Monitor(self.env, str_c,force=True, video_callable=lambda episode_id: episode_id%30==0)

        
        self.num_episodes = 2000
        self.num_iter = 500

        self.epsilon = 1
        self.batch_size = 32
        self.iteration_count = 0
    
        if(environment_name == 'CartPole-v0'):
            self.gamma = 0.95
        elif(environment_name == 'MountainCar-v0'):
            self.gamma = 1

        self.environment_name = environment_name
        self.input_shape = []
        self.obs_list = []
        self.iteration_count_list = []
        self.mean_reward_list = []
        self.test_final_count = 0

        self.fig3 = plt.figure()
        self.ax3 = self.fig3.gca()
        self.ax3.set_title('Average Return over episodes Plot - Test')

        self.fig5 = plt.figure()
        self.ax5 = self.fig5.gca()
        self.ax5.set_title('Reward per episodes Plot - Test')


    def epsilon_greedy_policy(self, Q, epsilon):
        # Creating epsilon greedy probabilities to sample from.

        action = np.argmax(Q)
        if np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()

        return action

    def greedy_policy(self, Q):
        # Creating greedy policy for test time.
        action = np.argmax(Q)
        return action

    def init_state(self):
        s = self.env.reset()
        return s

    def update_eps(self, eps, rTotal, num_success):
        if(self.environment_name == 'CartPole-v0'):
            if(eps > 0.01):
                eps = eps-0.0009
            else:
                eps = 0.1
        elif(self.environment_name == 'MountainCar-v0'):
            if(rTotal > -190):
                num_success +=1
            if num_success < 15:
                eps *= 0.9965
            else:
                eps = 0.1

            if(eps <0.1):
                eps = 0.1

        return eps, num_success

    def train_core(self, batch_s, tarQbatch):
        if(self.environment_name == 'CartPole-v0'):
            batch_s = np.reshape(batch_s, [32, 4])
            tarQbatch = np.reshape(tarQbatch, [32, 2])
        elif(self.environment_name == 'MountainCar-v0'):
            batch_s = np.reshape(batch_s, [32, 2])
            tarQbatch = np.reshape(tarQbatch, [32, 3])

        self.mainQN.model.fit(batch_s, tarQbatch, epochs=1, verbose=0)

    def get_Q_S(self, memory_batch):
        batch_s = [row[0] for row in memory_batch]
        batch_nexts = [row[3] for row in memory_batch]
        array1 = [np.reshape(inp, [1, inp.size]) for inp in batch_s]
        array2 = [np.reshape(inp, [1, inp.size]) for inp in batch_nexts]
        input = np.stack([array1, array2])
        input = np.reshape(input, [self.batch_size*2, inp.size])
        Qbatch_s = self.mainQN.model.predict(input)
        Qbatch_nexts = deepcopy(Qbatch_s[32:])
        tarQbatch = deepcopy(Qbatch_s[:32])

        for k in range(self.batch_size):
            if(memory_batch[k][4]==False):
                action = memory_batch[k][1]
                reward = memory_batch[k][2]
                tarQbatch[k][action] = reward + self.gamma*(np.max(Qbatch_nexts[k]))
            else:
                action = memory_batch[k][1]
                reward = memory_batch[k][2]
                tarQbatch[k][action] = reward

        return batch_s, tarQbatch, Qbatch_nexts

    def append_memory(self, s, action, reward, newState, done):
        
        if len(np.shape(s)) == 2:
            s = s[0]
        if len(np.shape(newState)) == 2:
            newState = newState[0]
        transition = [s, action, reward, newState, done]
        self.replay.append(transition)
        s = deepcopy(newState)

        return s


    def train(self, render ):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        # Variables init
        eps = self.epsilon
        rList = []
        num_success = 0
        g_time = time.time()
        flag_to_test = 0

        # Plot init
        fig1 = plt.figure()
        ax1 = fig1.gca()
        ax1.set_title('Per episode Return Plot')
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.set_title('Average Return over episodes Plot - Train')

        # Burn in memory
        self.burn_in_memory(self.environment_name)

        for i in range(self.num_episodes):
            s = self.init_state()
            rTotal = 0
            start_time = time.time()
            for j in range(self.num_iter):
                try:
                    self.iteration_count += 1
                    s = np.reshape(s, [1,s.size])
                    Q0 = self.mainQN.model.predict(s)
                    action = self.epsilon_greedy_policy(Q0,eps)
                    newState, reward, done, _ = self.env.step(action)
                    if render == 1:
                        self.env.render()

                    rTotal += reward

                    # Appending to Replay memory
                    s = self.append_memory(s, action, reward, newState, done)
                    # Sampling from Replay Memory
                    memory_batch = self.replay.sample_batch(self.batch_size)
                    # Forming the target Q and state to train
                    batch_s, tarQbatch, Qbatch_nexts = self.get_Q_S(memory_batch)

                    # Training the network
                    self.train_core(batch_s, tarQbatch)

                    if self.environment_name == 'CartPole-v0':
                        if fmod(self.iteration_count, 1000) == 0:
                            flag_to_test = 1
                    else:
                        if fmod(self.iteration_count, 10000) == 0:
                            flag_to_test = 1
        
                    # Exit if done
                    if done == True:
                        break

                except KeyboardInterrupt:
                    self.test_final(render)
                    self.save_final_weight()
                    sys.exit(0)

            end_time = time.time()
            print('Duration:{}'.format(end_time-start_time))

            if fmod(i, 30) == 0:
                self.mainQN.save_model_weights(int(i)/30)

            eps, num_success = self.update_eps(eps, rTotal, num_success)

            rList.append(rTotal)
            ax1.scatter(i, rTotal)
            ax2.scatter(i, np.mean(rList))
            print("Episode %s - Reward %s - eps %s" %(i,rTotal,eps))
            plt.pause(0.001)

            # Save the model weights every 30 episodes
            if(i%30==0):
                if not os.path.exists(os.path.join(self.environment_name, 'lnQ_Plot_train')):
                    os.makedirs(os.path.join(self.environment_name, 'lnQ_Plot_train'))

                file_name = 'Plot_instant_total_reward_' + str(g_time) + '.png'
                ax1.figure.savefig(os.path.join(self.environment_name, 'lnQ_Plot_train', file_name))
                file_name = 'Plot_average_total_reward_' + str(g_time) + '.png'
                ax2.figure.savefig(os.path.join(self.environment_name, 'lnQ_Plot_train', file_name))

            # Test once the flag is set
            if flag_to_test == 1:
                flag_to_test = 0
                self.test(render)
        
    def test(self, render, num_episodes = 20, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        rList = []
        g_time = time.time()

        if model_file is not None:
            self.mainQN.load_model_weights(model_file)

        for i in range(num_episodes):
            s = self.init_state()
            rTotal = 0
            done = False

            while not done:
                s = np.reshape(s, [1,s.size])

                Q0 = self.mainQN.model.predict(s)
                action = self.epsilon_greedy_policy(Q0,0.1)
                newState, reward, done, _ = self.env.step(action)
                if render == 1:
                    self.env.render()

                rTotal += reward
                s = deepcopy(newState)

            rList.append(rTotal)

        if self.environment_name == 'MountainCar-v0':
            if rTotal > -200:
                self.mainQN.save_model_weights('test_' + str(self.test_final_count))
                self.test_final_count += 1
                self.test_final(1, True)

        self.iteration_count_list.append(self.iteration_count)
        self.mean_reward_list.append(np.mean(rList))
        self.ax3.plot(self.iteration_count_list, self.mean_reward_list, 'k-*')
        plt.pause(0.001)

        if not os.path.exists(os.path.join(self.environment_name, 'lnQ_Plot_test')):
            os.makedirs(os.path.join(self.environment_name, 'lnQ_Plot_test'))
        
        file_name = 'Plot_average_total_reward_' + str(g_time) + '.png'
        self.ax3.figure.savefig(os.path.join(self.environment_name, 'lnQ_Plot_test', file_name))

    def test_final(self, render=1, done=False, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        rList = []
        iList = []
        self.test_final_count += 1

        g_time = time.time()

        num_episodes = 100
        if model_file is not None:
            self.mainQN.load_model_weights(model_file)

        while not done:
            action = self.env.action_space.sample()
            newState, reward, done, _ = self.env.step(action)

        for i in range(num_episodes):
            s = self.init_state()
            rTotal = 0
            done = False

            while not done:
                s = np.reshape(s, [1,s.size])

                Q0 = self.mainQN.model.predict(s)
                action = self.epsilon_greedy_policy(Q0,0.1)
                newState, reward, done, _ = self.env.step(action)
                if render == 1:
                    self.env.render()

                rTotal += reward
                s = deepcopy(newState)

            rList.append(rTotal)
            iList.append(i)

        self.ax5.plot(iList, rList, 'k-*')
        print('Mean Reward:{}'.format(np.mean(rList)))
        print('Std. dev in Reward:{}'.format(np.std(rList)))
        plt.pause(0.001)

        if not os.path.exists(os.path.join(self.environment_name, 'lnQ_Plot_test_final')):
            os.makedirs(os.path.join(self.environment_name, 'lnQ_Plot_test_final'))
        
        file_name = 'Plot_reward_over_episodes_' + str(g_time) + '.png'
        self.ax5.figure.savefig(os.path.join(self.environment_name, 'lnQ_Plot_test_final', file_name))


    def burn_in_memory(self,environment_name):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        if self.environment_name == 'MountainCar-v0':
            self.replay = Replay_Memory(environment_name, 10000, 500)
        else:
            self.replay = Replay_Memory(environment_name, 50000, 500)

    def save_final_weight(self):
        self.mainQN.save_model_weights('final')


def lnQ_main(args):
    environment_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)
  
    # You want to create an instance of the lnQ_Agent class here, and then train / test it.
    agentlnQ = lnQ_Agent(environment_name)

    if args.train == 1:
            agentlnQ.train(args.render)
            agentlnQ.test_final(args.render, True)
            agentlnQ.save_final_weight()
            sys.exit(0)

    else:
        if args.model_file is not None:
            agentlnQ.test_final(args.render, True, args.model_file)

