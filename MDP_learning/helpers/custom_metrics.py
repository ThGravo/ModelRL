from keras import backend as K


def COD(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1) / K.sum(K.square(y_pred - K.mean(y_pred, axis=-1)), axis=-1)


def Rsquared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=-1)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - SS_res / (SS_tot + K.epsilon())


def NRMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) / K.mean((K.max(y_true, axis=-1) - K.min(y_true, axis=-1)))
