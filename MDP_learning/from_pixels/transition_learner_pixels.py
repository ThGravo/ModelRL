from __future__ import division
import argparse
from PIL import Image
import numpy as np
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Input, LSTM, concatenate, Reshape, Lambda
import keras.backend as K
from rl.core import Processor




# input image dimensions
img_rows, img_cols = 105, 80
INPUT_SHAPE = (img_rows, img_cols)
WINDOW_LENGTH = 1

# IMAGE PRE-PROCESSING

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

# PARSING ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


class ModelLearner():
    def __init__(self, state_shape, action_shape, action_num, learning_rate=.001):
        # get size of state and action
        self.state_size = state_shape
        self.action_shape = action_shape
        self.action_num = action_num

        # These are hyper parameters
        self.learning_rate = learning_rate
        self.batch_size = 10
        self.net_train_epochs = 64

        # create main model and target model
        self.model = self.build_model(state_shape, action_shape)


    def build_model(self, state_shape, action_shape): # TODO: use state_shape and action_shape instead of fixed values
        #input_shape = (1, img_rows, img_cols)
        image_in = Input(shape=state_shape, name='image_input')
        action_in = Input(shape=action_shape, name='action_input')
        '''
        if K.image_dim_ordering() == 'tf':
            # (width, height, channels)
            input_perm = Permute((2, 3, 1), input_shape=state_shape)(image_in)
        elif K.image_dim_ordering() == 'th':
            # (channels, width, height)
            input_perm = Permute((1, 2, 3), input_shape=state_shape)(image_in)
        else:
            raise RuntimeError('Unknown image_dim_ordering.')
        
        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = Lambda(lambda x: x / 255.0)(input_perm)
        '''
        # Convolutional layers TODO: try pooling and dropout
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(image_in)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        conv_out = Flatten()(conv3)
        dense_out = Dense(512, activation='relu')(conv_out)
        dense_out = Reshape((1, 512))(dense_out)

        ## Recurrent encoding
        lstm_out = LSTM(512)(dense_out)

        ## Next-state predictor
        state_enc_in = concatenate([lstm_out, action_in], name='encoded_state_and_action')
        state_enc_out = Dense(512, activation='linear', name='predicted_next_state')(state_enc_in)

        # Q-values predictor
        q_out = Dense(self.action_num, activation='linear')(lstm_out)

        # Specify input and outputs for model
        model = Model(inputs=[image_in,action_in], outputs=[q_out, state_enc_out])
                      #outputs=[q_out, state_enc_out]) TODO: get multi-output to work
        print(model.summary())
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # Intermediate layer model to get predicted next states
        layer_name = 'predicted_next_state'
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        return model, intermediate_layer_model

def get_rollouts(env, batch_size, seed=None):
    rollout_limit = env.spec.timestep_limit
    states, actions, rewards = [], [], []
    for j in range(batch_size):
        env.seed(seed)
        s = env.reset()
        rollout_s, rollout_a, rollout_r = [], [], []
        for i in range(rollout_limit):
            a = env.action_space.sample()
            s1, r, done, _ = env.step(a)
            rollout_s.append(s)
            rollout_a.append(a)
            rollout_r.append(r)
            s = s1
            if done: break
        env.seed(None)
        states.append(np.array(rollout_s))
        actions.append(np.array(rollout_a))
        rewards.append(np.array(rollout_r))
    return np.array(states), np.array(actions), np.array(rewards)

'''
Sample rollouts
def sample(batch_size, trace_length, states, actions, rewards):
    sampled_traces_s, sampled_traces_a, sampled_traces_r = [],[],[]
    rollouts_to_sample = np.random.randint(0, high=len(states), size=batch_size)
    for rollout in rollouts_to_sample:
        point = np.random.randint(0, len(rollout) + 1 - trace_length)
        sampled_traces_s.append(states[point:point + trace_length])
        sampled_traces_a.append(actions[point:point + trace_length])
        sampled_traces_r.append(rewards[point:point + trace_length])
        sampled_traces_s = np.array(sampled_traces_s)
        sampled_traces_a = np.array(sampled_traces_a)
        sampled_traces_r = np.array(sampled_traces_r)
    return np.reshape(sampled_traces_s, [batch_size, trace_length]), np.reshape(sampled_traces_a, [batch_size, trace_length]), np.reshape(sampled_traces_r, [batch_size, trace_length])
'''

# training settings

epochs = 1000 # number of training batches
batch_size = 10 # number of rollouts per training batch
rollout_limit = env.spec.timestep_limit # max rollout length
discount_factor = 1.00 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.001 # you know this by now
early_stop_loss = 0 # stop training if loss < early_stop_loss, 0 or False to disable

# train policy network

if __name__ == "__main__":
    for env_name in ['Breakout-v0']:  # ['LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']:
        env = gym.make(env_name)
        # get size of state and action from environment
        state_shape = env.observation_space.shape
        action_shape = (1,) # TODO: get shape from environment. something like env.action.space.shape?
        num_discrete_actions = env.action_space.n
        agent = ModelLearner(state_shape,action_shape, num_discrete_actions) # TODO: prepro of frames
        model, intermediate_layer_model = agent.model
        states, actions, rewards = [], [], []  # lists of rollouts
        for epoch in range(epochs):
            # fill batch with rollouts
            states, actions, rewards = get_rollouts(env, batch_size, seed=epoch)
            '''       
                # sample traces
                sampled_traces_states = []
                sampled_traces_actions = []
                for i in range(len(states)):
                    s , a = states[i], actions[i]
                    if len(states[i]) > trace_length:
                        point = np.random.randint(0, len(s) + 1 - trace_length)
                        sampled_traces_states.append(s[point:point + trace_length])
                        sampled_traces_actions.append(a[point:point + trace_length])
                sampled_traces_states = np.concatenate(sampled_traces_states)
                sampled_traces_actions = np.concatenate(sampled_traces_actions)
            '''
            get_layer_output = K.function([model.layers[0].input, model.layers[8].input],
                                              [model.layers[11].output])
            layer_output = get_layer_output([states, actions])

            inp = model.input  # input placeholder
            outputs = [layer.output for layer in model.layers]  # all layer outputs
            functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

            # Testing
            test = np.random.random(input_shape)[np.newaxis, ...]
            layer_outs = functor([test, 1.])


            target_states = intermediate_layer_model.predict([states,actions])
            # do the model fit
            agent.fit([actions,states],

                            batch[:, -model.state_size - 1:-1],
                            batch_size=batch_size,
                            validation_split=0.1,
                            verbose=0)

            print('MSE: {}'.format(model.evaluate(env)))





