import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Permute, Input, LSTM, concatenate, Reshape
from keras.callbacks import TensorBoard, Callback
from keras.utils import to_categorical
from keras.losses import mse, mape
import keras.backend as K
import time
import os
from MDP_learning.helpers.custom_metrics import COD, NRMSE, Rsquared


# and define your own Callback

class MyCallback(Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.alpha, max(K.get_value(self.alpha) / 1.1, .01))
        # K.set_value(self.beta, K.get_value(self.beta) * 1.01)
        # logger.info("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))
        print("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))


# then pass it into fit

# model.fit( ..., callbacks=[MyCallback(alpha, beta)], ...)


def train_icm(cfg, dqn, nb_actions, do_load_dqn_weights=True):
    state_size = cfg.input_shape

    # Inverse Model learner network
    action_in = Input(shape=(nb_actions,), name='action_input')

    # feature encoding: phi1, phi2: [None, LEN]
    size = 256
    # Defined a shared model
    image_in = Input(shape=cfg.input_shape, name='main_input')
    input_perm = Permute((2, 3, 1), input_shape=cfg.input_shape)(image_in)
    conv1 = Conv2D(32, (8, 8), activation="relu", strides=(4, 4), name='conv1')(input_perm)
    conv2 = Conv2D(64, (4, 4), activation="relu", strides=(2, 2), name='conv2')(conv1)
    conv3 = Conv2D(64, (3, 3), activation="relu", strides=(1, 1), name='conv3')(conv2)
    flat_feat = Flatten(name='flat_feat')(conv3)
    dense_out = Dense(512, activation='relu')(flat_feat)
    shared_conv_model = Model(image_in, dense_out, name="shared_flat_feat")

    # Define two input image seq
    image0_in = Input(shape=cfg.input_shape, name='state_t0')
    image1_in = Input(shape=cfg.input_shape, name='state_t1')
    encoded_0 = shared_conv_model(image0_in)
    encoded_1 = shared_conv_model(image1_in)

    # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
    cat_phi = concatenate([encoded_0, encoded_1], axis=-1)
    g1 = Dense(size, activation='relu', name="g1")(cat_phi)
    a_out = Dense(nb_actions, activation='softmax', name='imodel')(g1)

    # forward model: f(phi1,asample) -> phi2
    feat_act = concatenate([encoded_0, action_in], axis=-1)
    fc = Dense(size, activation='relu')(feat_act)  # encoded_0._keras_shape[1]
    state_pred = Dense(encoded_0._keras_shape[1], activation='linear', name='fmodel')(fc)

    def custom_loss(y_true, y_pred):
        return mse(encoded_1, y_pred)

    def custom_metric(y_true, y_pred):
        return mse(encoded_1[:, :y_pred.shape[1]], y_true)

    alpha = K.variable(5000)
    beta = K.variable(1)
    im_model = Model(inputs=[image0_in, image1_in, action_in], outputs=[a_out, state_pred])
    im_model.compile(optimizer='nadam', loss=['sparse_categorical_crossentropy', custom_loss],
                     metrics=['sparse_categorical_accuracy', mape, custom_metric], loss_weights=[alpha, beta])

    if do_load_dqn_weights:
        shared_conv_model.load_weights(cfg.filename, by_name=True, skip_mismatch=True)
    '''
    seq_shape = (007, 007)

    # Defined a shared model, 2 LSTM layers
    input_seq = Input(seq_shape)
    lstm_1 = LSTM(10, return_sequences=True)(input_seq)
    lstm_2 = LSTM(10)(lstm_1)
    shared_lstm = Model(input_seq, lstm_2, name="shared_lstm")

    # Define two input sequences
    seq_1 = Input(seq_shape)
    seq_2 = Input(seq_shape)

    encoded_1 = shared_lstm(seq_1)
    encoded_2 = shared_lstm(seq_2)

    # Let's do classification for fun
    num_classes = 3
    cat_phi = keras.layers.concatenate([encoded_1, encoded_2], axis=-1)
    softmax = Dense(num_classes, activation="softmax")(cat_phi)

    m = Model([seq_1, seq_2], softmax)

    from keras.utils import plot_model
    plot_model(m, to_file='model.png')
    '''
    '''
    tweet_a = Input(shape=(280, 256))
    tweet_b = Input(shape=(280, 256))
    
    # This layer can take as input a matrix
    # and will return a vector of size 64
    shared_lstm = LSTM(64)

    # When we reuse the same layer instance
    # multiple times, the weights of the layer
    # are also being reused
    # (it is effectively *the same* layer)
    encoded_a = shared_lstm(tweet_a)
    encoded_b = shared_lstm(tweet_b)

    # We can then concatenate the two vectors:
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

    # And add a logistic regression on top
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    # We define a trainable model linking the
    # tweet inputs to the predictions
    model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit([data_a, data_b], labels, epochs=10)
    '''
    """"""

    print(im_model.summary())

    log_string = 'dyna_{}_nb_actions{}_loadDQNwgths{}-{}'.format(cfg.env_name, nb_actions, do_load_dqn_weights,
                                                                time.time())
    print('logging to {}'.format(log_string))
    ####################################################################################################################

    data_size = dqn.memory.observations.length
    chunk_size = int(max(min(data_size / 2, 10000), data_size / 100))
    n_rounds = 2 * int(data_size / chunk_size) + 1  # go through data 3 times
    for ii in range(n_rounds):
        print("{} of {} n_rounds".format(ii, n_rounds))
        experiences = dqn.memory.sample(chunk_size)
        # Start by extracting the necessary parameters
        state0_seq = []
        state1_seq = []
        reward_seq = []
        action_seq = []
        terminal1_seq = []

        for e in experiences:
            state0_seq.append(e.state0)
            state1_seq.append(e.state1)
            reward_seq.append(e.reward)
            action_seq.append(e.action)
            terminal1_seq.append(e.terminal1)

        state0_seq = dqn.process_state_batch(state0_seq)
        state1_seq = dqn.process_state_batch(state1_seq)
        hstate1_seq = shared_conv_model.predict_on_batch(state1_seq)
        reward_seq = np.array(reward_seq)
        action_seq = np.array(action_seq)
        terminal1_seq = np.array(terminal1_seq)

        # one hot encode
        actions_encoded = to_categorical(action_seq, num_classes=nb_actions)

        if False:  # Debug
            with open("{}DataStats.txt".format(cfg.env_name), "w") as text_file:
                print("Var: {}".format(np.var(next_state)), file=text_file)
                print("Min: {}".format(np.mean(np.min(next_state, axis=0))), file=text_file)
                print("Max: {}".format(np.mean(np.max(next_state, axis=0))), file=text_file)
                print("Min: {}".format(np.min(next_state, axis=0)), file=text_file)
                print("Max: {}".format(np.max(next_state, axis=0)), file=text_file)

        im_model.fit([state0_seq, state1_seq, actions_encoded], [action_seq, hstate1_seq],
                     validation_split=0.2, verbose=1, epochs=cfg.ml_model_epochs, callbacks=[MyCallback(alpha, beta),
                                                                                             TensorBoard(
                                                                                                 log_dir='./invmod_logs/{}/{}'.format(
                                                                                                     os.path.splitext(
                                                                                                         cfg.filename)[
                                                                                                         0],
                                                                                                     log_string))])
    return im_model
    # #######################################################################################################################
