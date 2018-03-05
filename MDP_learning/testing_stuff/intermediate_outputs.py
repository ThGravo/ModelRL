from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten
from keras import backend as K

import numpy as np

trX = np.linspace(-1, 1, 101)
print(trX.shape)
trY = 3 * trX + np.random.randn(*trX.shape) * 0.33

input_layer = Input(shape=(1,), name='input_layer')
dense_1 = Dense(64, activation='relu', name='d1')(input_layer)
dense_2 = Dense(1, activation='relu', name='d2')(dense_1)
model = Model(inputs=[input_layer], outputs=[dense_2])

print(model.summary())

model.compile(optimizer='sgd', loss='mse')

model.fit(trX, trY, nb_epoch=200, verbose=1)


print("INTERMEDIATE")
layer_name = 'd1'
layer_name2 = 'd2'
intermediate_layer_model = Model(inputs=[input_layer],
                                 outputs=model.get_layer(layer_name).output)
intermediate_layer_model.predict(trX)
intermediate_layer_model2 = Model(inputs=[input_layer],
                                 outputs=model.get_layer(layer_name2).output)
intermediate_layer_model2.predict(trX)
