from keras import Input, Model
from keras.layers import Permute, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def setupDQN(cfg, nb_actions, processor):
    image_in = Input(shape=cfg.input_shape, name='main_input')
    input_perm = Permute((2, 3, 1), input_shape=cfg.input_shape)(image_in)
    conv1 = Conv2D(32, (8, 8), activation="relu", strides=(4, 4), name='conv1')(input_perm)
    conv2 = Conv2D(64, (4, 4), activation="relu", strides=(2, 2), name='conv2')(conv1)
    conv3 = Conv2D(64, (3, 3), activation="relu", strides=(1, 1), name='conv3')(conv2)
    conv_out = Flatten(name='flat_feat')(conv3)
    dense_out = Dense(512, activation='relu')(conv_out)
    q_out = Dense(nb_actions, activation='linear')(dense_out)
    model = Model(inputs=[image_in], outputs=[q_out])
    print(model.summary())
    # hstate_size = int(np.prod(conv3.shape[1:]))

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=cfg.memory_limit, window_length=cfg.WINDOW_LENGTH)

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=cfg.nb_steps_annealed_policy)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=cfg.nb_steps_warmup_dqn_agent, gamma=.99,
                   target_model_update=cfg.target_model_update_dqn_agent,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    return dqn


def trainDQN(cfg, env, dqn):
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    callbacks = [ModelIntervalCheckpoint(cfg.checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(cfg.log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=cfg.nb_steps_dqn_fit, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(cfg.weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    # dqn.test(env, nb_episodes=1, visualize=False)