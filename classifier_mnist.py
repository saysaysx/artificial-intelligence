import keras.layers
import matplotlib.pyplot as plt
import tensorflow.keras as krs
import numpy

from keras.datasets import mnist
import tensorflow.keras.utils as ut

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

nw = 28
nh = 28


(trainx, trainy), (testx, testy) = mnist.load_data()
all_image = (trainx/255.0-0.5)*1.999
all_image = numpy.expand_dims(all_image, axis=3)

all_out = ut.to_categorical(trainy)

testx = (testx/255.0-0.5)*1.999
testy = ut.to_categorical(testy)


desc_input = krs.layers.Input(shape=(nw,nh,1))
lay = krs.layers.Conv2D(32, (3, 3), strides = (2,2), activation='relu', padding='same')(desc_input)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(64, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(128, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(256, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
lay = krs.layers.Flatten()(lay)
lay_out = krs.layers.Dense(10, activation="softmax", name='den4')(lay)

classificator = krs.Model(desc_input, lay_out)
classificator.trainable = True
classificator.compile(loss='binary_crossentropy', optimizer=krs.optimizers.Adam(learning_rate = 0.0002),
                      metrics=['accuracy'])

classificator.fit(all_image, all_out, batch_size = 8000 ,epochs = 150, validation_data=(testx,testy))

classificator.save("./out/class_mnist.h5")
