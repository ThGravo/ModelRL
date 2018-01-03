from collections import deque
import numpy as np
import gym


class SynthEnv(object):
    def __init__(self, tmodel, conv_model, real_env, processor, sequence_len, WINDOW_LENGTH):
        self.tmodel = tmodel
        self.conv_model = conv_model
        self.real_env = real_env
        self.processor = processor
        self.seq_len = sequence_len
        self.WINDOW_LENGTH = WINDOW_LENGTH
        self.action_space = real_env.action_space
        # self.observation_space = gym.spaces.Box(-10, 10, (int(np.prod(conv3.shape[1:])),))
        self.state_seq, self.action_seq = self.init_state()

    def init_state(self):
        state_seq = deque(maxlen=self.seq_len)
        act_seq = deque(maxlen=self.seq_len)  # TODO should be just one action
        self.real_env.reset()

        images = []
        for _ in range(self.seq_len + self.WINDOW_LENGTH):
            act_seq.append(self.real_env.action_space.sample())
            obs, rw, dn, info = self.real_env.step(act_seq[-1])
            obs = self.processor.process_observation(obs)
            images.append(obs)

        for i in range(self.seq_len):
            state_seq.append(
                self.conv_model.predict(
                    np.expand_dims(np.array(images[i:i + self.WINDOW_LENGTH]), axis=0)
                )
            )

        return state_seq, act_seq

    def step(self, action):
        # TODO append action before?
        self.action_seq.append(action)
        # reshape
        ssq = np.rollaxis(np.array(self.state_seq), 1)
        asq = np.expand_dims(np.expand_dims(np.array(self.action_seq), axis=0), axis=2)
        next_state, reward, done = self.tmodel.predict([ssq, asq])
        self.state_seq.append(next_state)
        # unwrap and add empty info
        return next_state[0], float(reward[0, 0]), bool(done[0, 0] > .5), {}

    # TODO done might never occur in unseen territory
    # TODO check timing t->t+1

    def reset(self):
        self.state_seq, self.action_seq = self.init_state()
        return self.state_seq[-1].flatten()
