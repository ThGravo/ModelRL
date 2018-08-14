import numpy as np


class AtariConfig:
    def __init__(self, env_name='PongDeterministic-v4', weights=None):
        self.INPUT_SHAPE = (84, 84)
        self.WINDOW_LENGTH = 4
        self.nb_steps_dqn_fit = 177700#0
        self.nb_steps_warmup_dqn_agent = int(max(0, np.sqrt(self.nb_steps_dqn_fit))) * 42 + 42  # 50000
        self.target_model_update_dqn_agent = int(max(0, np.sqrt(self.nb_steps_dqn_fit))) * 8 + 8  # 10000
        self.memory_limit = int(self.nb_steps_dqn_fit / 2)  # 1000000
        self.nb_steps_annealed_policy = int(self.nb_steps_dqn_fit / 2)  # 1000000
        self.ml_model_epochs = 30

        # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
        self.input_shape = (self.WINDOW_LENGTH,) + self.INPUT_SHAPE

        # saving
        self.env_name = env_name
        self.weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
        self.checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
        self.log_filename = 'dqn_{}_log.json'.format(env_name)

        # loading
        self.filename = 'dqn_{}_weights.h5f'.format(env_name)
        if weights:
            self.filename = weights