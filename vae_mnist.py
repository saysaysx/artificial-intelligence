import matplotlib.pyplot as plt
import tensorflow.keras as krs


import PIL
import numpy
import os

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

nw = 28
nh = 28

from keras.datasets import mnist
(trainx, trainy), (testx, testy) = mnist.load_data()

all_image = (trainx/255-0.5)*1.999
num_hide = 128

print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())


# слой свертки для дискриминатора
def conv(filt,size,x):
    y = krs.layers.Conv2D(filters = filt, strides=(2,2), kernel_size = (size,size),padding='same') (x)

    lay = krs.layers.BatchNormalization()(y)
    return lay
# слой деконволюции для генератора
def dconv(filt,size,x, strides=2):
    y = krs.layers.UpSampling2D(size=strides)(x)
    y = krs.layers.Conv2D(filters = filt,kernel_size=size,padding='same') (y)
    lay = krs.layers.BatchNormalization()(y)
    return lay






encoder_input = krs.layers.Input(shape = (nh,nw,1))
lay = conv(32,5,encoder_input)
lay = conv(64,4,lay)
lay = conv(32,3,lay)

lay = krs.layers.Flatten()(lay)
lay = krs.layers.Dense(num_hide) (lay)



z_mean = krs.layers.Dense(num_hide, activation='linear', name = 'z_mean') (lay)
z_std = krs.layers.Dense(num_hide, activation='relu', name = 'z_std') (lay)

noise_input = krs.layers.Input(shape  =(num_hide,))
noise = krs.layers.GaussianNoise(stddev = 1) (noise_input)


mult = krs.layers.Multiply()([z_std,noise])
out = krs.layers.Add() ([mult,z_mean])

def vae_layer(args):
    z_mean, z_std = args
    z_std_sq = z_std*z_std
    z_mean_sq = z_mean*z_mean
    all = -(1+krs.backend.log(z_std_sq+1e-17)-z_std_sq-z_mean_sq)*0.5
    return all

def vae_loss(y_true,x_pred):
    x = krs.backend.sum(x_pred, axis=-1)/num_hide

    return krs.backend.mean(x)

out_loss = krs.layers.Lambda(vae_layer) ([z_mean,z_std])


encoder = krs.Model(encoder_input, outputs = [z_mean,z_std],name = 'encoder')

decoder_input = krs.layers.Input(shape = (num_hide,))
lay = krs.layers.Dense(7 * 7 * 128, activation='relu') (decoder_input)
lay = krs.layers.Reshape(target_shape=(7,7,128)) (lay)
lay = dconv(128,3,lay)
lay = dconv(64,4,lay)

lay = krs.layers.Conv2D(filters = 1,kernel_size=5,padding='same',activation='tanh') (lay)
decoder = krs.Model(decoder_input, outputs = lay, name = 'decoder')

modelall = decoder(out)
vae = krs.Model([encoder_input,noise_input], [modelall,out_loss], name = 'vae')






vae.compile(loss = ['mean_squared_error',vae_loss],optimizer=krs.optimizers.Adam(lr=0.0009),
			metrics=['accuracy'])

krs.utils.plot_model(vae, to_file='./out/modelvae.png', show_shapes=True)
krs.utils.plot_model(encoder, to_file='./out/encoder.png', show_shapes=True)
krs.utils.plot_model(decoder, to_file='./out/decoder.png', show_shapes=True)
x = all_image



x_noise = numpy.zeros((x.shape[0],num_hide))
y_noise = numpy.zeros((x.shape[0]))


vae.load_weights("./out/vae_weights7.h5")
vae.fit([all_image,x_noise],[all_image,y_noise],batch_size=1000,epochs=2070)
vae.save_weights("./out/vae_weights7.h5")
nex = 5
pred = vae.predict([all_image[0:nex],x_noise[0:nex]])
print(pred[0].shape)
print(pred[1].shape)

pred_enc = encoder.predict(all_image[0:nex])
print(pred_enc[0].max())
print(pred_enc[0].min())
print(pred_enc[1].max())
print(pred_enc[1].min())
rx  = numpy.random.randn(nex,num_hide)
inp_dec = rx#pred_enc[0]+pred_enc[1]*rx
pred_dec = decoder.predict(inp_dec)

fig = plt.figure(figsize= (7,7))
ax = [[],[],[],[]]
for i in range(4):
    ax[i] = fig.add_subplot(2,2,i+1)
    ax[i].imshow(pred_dec[i][:,:,0:3]+1)

encoder.save("./out/vae_encoder.h5")
decoder.save("./out/vae_decoder.h5")


plt.show()
