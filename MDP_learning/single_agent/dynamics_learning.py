from collections import deque
from keras.callbacks import TensorBoard
import gym
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN, SimpleFill, SoftImpute, MICE, IterativeSVD, NuclearNormMinimization, MatrixFactorization, \
    BiScaler
from MDP_learning.single_agent.preprocessing import standardise_memory, make_mem_partial_obs, setup_batch_for_RNN, \
    impute_missing
from MDP_learning.single_agent.networks import build_regression_model, build_recurrent_regression_model, build_dmodel
from time import time
import random

from MDP_learning.helpers.logging_model_learner import LoggingModelLearner


# A neural network dynamics model learner under partial observability

class ModelLearner(LoggingModelLearner):
    def __init__(self, env_name, observation_space, action_space, data_size=200000, epochs=100, learning_rate=.001,
                 tmodel_dim_multipliers=[1,1], tmodel_activations=('relu', 'relu'), sequence_length=0,
                 partial_obs_rate=0.0):
        from collections import namedtuple
        Spec = namedtuple('Spec', 'id')
        Myenv = namedtuple('Myenv', ['spec'])
        t = Myenv(Spec(env_name))  # HACK to get the name injected there
        super().__init__(t, sequence_length, out_dir_add='po_rate{}'.format(partial_obs_rate))

        # get size of state and action from environment
        self.state_size = sum(observation_space.shape)
        self.action_size = 1
        self.action_num = 0
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_num = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            self.action_size = sum(action_space.shape)
        else:
            raise ValueError("The action_space is of type: {} - which is not supported!".format(type(action_space)))

        # These are hyper parameters
        self.learning_rate = learning_rate
        self.net_train_epochs = epochs
        self.partial_obs_rate = partial_obs_rate

        # create replay memory using deque
        self.data_size = data_size
        self.memory = deque(maxlen=self.data_size)

        # tmodel_dim_multipliers was used to increase the number of units
        # per layer as a function of the state size in the environment
        # currently it just affects the number of layers used in the network

        if self.useRNN:
            self.tmodel = build_recurrent_regression_model(self.state_size + self.action_size, self.state_size,
                                                           lr=learning_rate,
                                                           dim_multipliers=tmodel_dim_multipliers,
                                                           activations=tmodel_activations)
            self.seq_mem = deque(maxlen=self.sequence_length)
        else:
            self.tmodel = build_regression_model(self.state_size + self.action_size, self.state_size,
                                                 lr=learning_rate,
                                                 dim_multipliers=tmodel_dim_multipliers,
                                                 activations=tmodel_activations)

        #This code can be used to create networks that learn to predict
        #rewards and whether the next state is terminal or not.
        
                
        self.rmodel = build_regression_model(self.state_size, 1, lr=learning_rate,
                                             dim_multipliers=(4, 4),
                                             activations=('sigmoid', 'sigmoid'))
        self.dmodel = build_dmodel(self.state_size)

        # logging config of the network
        self.models = [self.tmodel, self.rmodel, self.dmodel]
        self.save_model_config()

    # get action from model using random policy
    def get_action(self, state, environment):
        return environment.action_space.sample()

    # filling up memory of transitions
    def refill_mem(self, environment):
        state = environment.reset()
        self.memory.clear()
        for i in range(self.data_size):
            action = self.get_action(state, environment)
            next_state, reward, done, info = environment.step(action)
            self.memory.append(np.hstack((state, action, reward, next_state, done * 1)))
            if done:
                state = environment.reset()
            else:
                state = next_state

    # defines the training process
    def train_models(self, minibatch_size=512):

        memory_arr = np.array(self.memory)
        
        '''
        put this code back if you want to corrupt
        the agent's memory, and then run an imputation
        on the corrupted memory
        
        if self.partial_obs_rate > 0:
            # creating missing state feature values
            make_mem_partial_obs(memory_arr, self.state_size, self.partial_obs_rate)
            print("Memory size:")
            print(memory_arr.size)
            print("Proportion of missing values:")
            print(np.isnan(memory_arr).sum() / memory_arr.size)
            # imputing missing values
            impute_missing(memory_arr, self.state_size, MICE)
        '''
        
        #normalizing the data values to [0,1]
        standardise_memory(memory_arr, self.state_size, self.action_size)
        batch_size = len(memory_arr)
        minibatch_size = min(minibatch_size, batch_size)

        if self.useRNN:
            t_x, t_y = setup_batch_for_RNN(memory_arr, self.sequence_length, self.state_size, self.action_size)
        else:
            t_x = memory_arr[:, :self.state_size + self.action_size]
            t_y = memory_arr[:, -self.state_size - 1:-1] - memory_arr[:, :self.state_size]

        self.tmodel.fit(t_x, t_y,
                        batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        validation_split=0.1,
                        callbacks=self.Ttensorboard,
                        verbose=1)
        '''
        #This code can be used for reward and terminal state prediction.
        self.rmodel.fit(batch[:, :self.state_size],
                        batch[:, self.state_size + self.action_size],
                        batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        validation_split=0.1,
                        callbacks=self.Rtensorboard, verbose=0)

        self.dmodel.fit(batch[:, :self.state_size],
                        batch[:, -1],
                        batch_size=minibatch_size,
                        epochs=self.net_train_epochs,
                        validation_split=0.1,
                        callbacks=self.Dtensorboard, verbose=0)
        '''
    #This function can be used to create new simulated experiences using the
    #trained dynamics, reward and terminal models
     
    def step(self, state, action):
        batch_size = 1
        state = np.reshape(state, [1, self.state_size])
        action = np.reshape(action, [1, self.action_size])

        if self.useRNN:
            self.seq_mem.append(np.hstack((state, action)))
            if len(self.seq_mem) is self.sequence_length:
                seq = np.array(self.seq_mem)
                next_state = self.tmodel.predict(np.rollaxis(seq, 1), batch_size)
            else:
                next_state = state
        else:
            next_state = self.tmodel.predict(np.hstack((state, action)), batch_size)
        reward = self.rmodel.predict(state, batch_size)
        done = self.dmodel.predict(state, batch_size)

        return next_state[0], float(reward[0, 0]), bool(done[0, 0] > .8)


    # run the training
    def run(self, environment, rounds=1):
        for e in range(rounds):
            self.refill_mem(environment)
            self.train_models()


if __name__ == "__main__":
    # ['Ant-v1', 'LunarLander-v2', 'BipedalWalker-v2', FrozenLake8x8-v0, 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:"Pong-ram-v4"
    import pickle

    # either just pickle
    if False:  # then save
        for env_name in ['Swimmer-v1', 'BipedalWalker-v2', 'Ant-v1', 'LunarLander-v2', 'Hopper-v1', 'HalfCheetah-v1',
                         'Walker2d-v1', 'Humanoid-v1', 'HumanoidStandup-v1', 'Reacher-v1', 'InvertedDoublePendulum-v1',
                         'InvertedPendulum-v1']:
            env = gym.make(env_name)
            observation_space = env.observation_space
            action_space = env.action_space
            with open('{}_observation_space.pickle'.format(env_name), 'wb') as f:
                pickle.dump(observation_space, f)
            with open('{}_action_space.pickle'.format(env_name), 'wb') as f:
                pickle.dump(action_space, f)

    else:
        # or load pickle and run a training
        env_name = "Swimmer-v1"
        with open('{}_observation_space.pickle'.format(env_name), 'rb') as f:
            observation_space = pickle.load(f)
        with open('{}_action_space.pickle'.format(env_name), 'rb') as f:
            action_space = pickle.load(f)
        ML = ModelLearner(env_name, observation_space, action_space, partial_obs_rate=0.25, sequence_length=0)
        mem0 = np.load('../save_memory1/{}IMPUTED0.25round0.npy'.format(env_name))
        mem1 = np.load('../save_memory1/{}IMPUTED0.25round1.npy'.format(env_name))
        mem2 = np.load('../save_memory1/{}IMPUTED0.25round2.npy'.format(env_name))
        mem3 = np.load('../save_memory1/{}IMPUTED0.25round3.npy'.format(env_name))
        mem4 = np.load('../save_memory1/{}IMPUTED0.25round4.npy'.format(env_name))
        mem = np.vstack((mem0,mem1,mem2,mem3,mem4))

        print(mem.shape)

        standardise_memory(mem, ML.state_size, ML.action_size)
        ML.memory = mem
        ML.train_models()
