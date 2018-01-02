from keras import backend as K


def COD(y_true, y_pred):
    sse = K.sum(K.square(y_pred - y_true), axis=0)
    scale = K.sum(K.square(y_pred - K.mean(y_pred, axis=0)), axis=0)
    return K.mean(sse / (scale + K.epsilon()), axis=-1)


def Rsquared(y_true, y_pred):
    sse = K.sum(K.square(y_true - y_pred), axis=0)
    tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    return K.mean(1.0 - sse / (tot + K.epsilon()), axis=-1)


def NRMSE(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=0)
    scale = K.max(y_true, axis=0) - K.min(y_true, axis=0)
    return K.mean(K.sqrt(mse) / (scale + K.epsilon()), axis=-1)
