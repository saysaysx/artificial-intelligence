# WGAN with clips comments and проверкой на условие липшица
import matplotlib.pyplot as plt
import tensorflow.keras as krs


import numpy

import tensorflow as tf
import pandas as pd

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


import gc
from keras.datasets import mnist

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def trainable(model, flag):
    model.trainable = flag
    for l in model.layers:
        l.trainable = flag
    return


def model_wloss_gan(input):

    def wloss(y_true,x_pred):
        return krs.backend.mean(y_true*x_pred)
    return wloss

# учет ограничения по Липшицу в функции потерь со штрафом
def model_wloss_descr(input):
    def wloss(y_true,x_pred):
        xval = krs.backend.mean(krs.backend.abs(input[:,None] - input[None,:]))
        fval = krs.backend.mean(krs.backend.abs(x_pred[:,None] - x_pred[None,:]))
        return krs.backend.mean(y_true*x_pred)+(krs.backend.relu(fval-xval))*0.85
    return wloss



nw = 28
nh = 28
num_hide = 196



(trainx, trainy), (testx, testy) = mnist.load_data()
all_image = (trainx/255.0-0.5)*1.999
all_image = numpy.expand_dims(all_image, axis=3)

print(all_image.shape)


desc_input = krs.layers.Input(shape=(nw,nh,1))
lay = krs.layers.Conv2D(32, (3, 3), strides = (2,2), activation='relu', padding='same')(desc_input)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(64, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(128, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(256, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
lay = krs.layers.Flatten()(lay)
lay_out = krs.layers.Dense(1, activation="linear", name='den4')(lay)


descriminator = krs.Model(desc_input, lay_out)
descriminator.trainable = True
descriminator.compile(loss=model_wloss_descr(descriminator.input), optimizer=tf.keras.optimizers.legacy.RMSprop(0.0001), metrics=['accuracy'])
gen_input = krs.layers.Input(shape=(num_hide,))
lay = krs.layers.Dense(128*7*7)(gen_input)
lay = krs.layers.Reshape(target_shape=(7,7,128))(lay)
lay = krs.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(lay)
lay = krs.layers.UpSampling2D(size=(2,2))(lay)
lay = krs.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(lay)
lay = krs.layers.UpSampling2D(size=(2,2))(lay)
lay_out = krs.layers.Conv2D(1, (3, 3), activation='tanh', padding='same')(lay)

generator = krs.Model(gen_input,lay_out)

krs.utils.plot_model(descriminator, to_file='./out/descriminator.png', show_shapes=True)

descriminator.trainable = False
gan = descriminator(generator.layers[-1].output)
gan_model = krs.Model(inputs=generator.layers[0].input, outputs=gan)

# binary_crossentropy
gan_model.compile(loss = model_wloss_gan(generator.layers[-1].output), optimizer=krs.optimizers.legacy.RMSprop(0.0001),
                  metrics=['accuracy'])

# binary_crossentropy

krs.utils.plot_model(gan_model, to_file='./out/gan_model.png', show_shapes=True)
# можно увеличить количество эпох обучения до 50 тыс
n_learn = 6000
n_batch = 32

ax = [[], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []]
classificator = krs.models.load_model("./out/class_mnist.h5")

error = []

n_batch_check = 4000
ones_c = numpy.ones((n_batch_check, 1))   -1e-13



ones = numpy.ones((n_batch,1))
neg_ones = -numpy.ones((n_batch,1))
n_critic = 1



for i_learn in range(n_learn):
    for j in range(n_critic):
        indexes = numpy.random.randint(0,all_image.shape[0],n_batch)
        real_image = all_image[indexes]
        rx  = numpy.random.randn(n_batch,num_hide)
        fake_image = generator.predict(rx)

        trainx = numpy.concatenate([real_image,fake_image])
        trainy = numpy.concatenate([neg_ones,ones])
        dloss, _ = descriminator.train_on_batch(trainx,trainy)
        # стандартный подход на основе clipweights
        #for l in descriminator.layers:
        #    weights = l.get_weights()
        #    weights = [numpy.clip(w, -0.01, 0.01) for w in weights]
        #    l.set_weights(weights)
    rx  = numpy.random.randn(n_batch,num_hide)
    gloss, _ = gan_model.train_on_batch(rx,neg_ones)



    if (i_learn % 1000 == 0):
        print(f"index {i_learn} dloss {dloss}  gloss {gloss}  ")


        rx = numpy.random.randn(n_batch_check,num_hide)


        #ans_gan = generator.predict(inp_dec)
        fake_image = generator.predict(rx)
        # проверка генерируемых образов обученным классификатором
        ans_cls = classificator.predict_on_batch(fake_image)
        #print(ans_cls.shape)
        #print(ans_cls[0:5])
        #print(ans_cls.argmax(axis=1))
        # рисуем гистограмму для сгенерированных образов
        arr = numpy.bincount(ans_cls.argmax(axis=1))
        print(arr)
        parr = arr/arr.sum()+1e-20
        H = (parr*numpy.log(1/parr)).sum()
        error.append([i_learn, H ])
        print(f"entropy {H}")

        if (i_learn % 2000 == 0 and i_learn > 4000):
            descriminator.save_weights("./out/model_descriminator13.h5")
            generator.save_weights("./out/model_generator13.h5")

        rx = numpy.random.normal(0, 1, (16, num_hide))#numpy.random.randn(16,num_hide)
        #inds = numpy.random.randint(0, n_batch, 16)
        inp_dec = rx #pred_enc[0][inds] + pred_enc[1][inds] * rx[inds]
        pred_dec = generator.predict(inp_dec)


        fig = plt.figure(figsize = (5,5))
        for i in range(16):
            ax[i] = fig.add_subplot(4, 4, i + 1)
            ax[i].imshow((pred_dec[i][:, :, 0]+1)*0.5)
        plt.savefig(f'./out/dig_{i_learn}.png', dpi=200)

error  = numpy.array(error)
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(error[:,0],error[:,1])
ax1.set_ylabel('Entropy')
ax1.set_xlabel('Number of iteration')

rx = numpy.random.randn(n_batch_check,num_hide)
fake_image = generator.predict(rx)
# проверка распознавания генерируемых цифр классификатором
# проверка на равномерность
ans_cls = classificator.predict_on_batch(fake_image)
arr = numpy.bincount(ans_cls.argmax(axis=1))


ax2.bar([0,1,2,3,4,5,6,7,8,9], arr)
ax2.set_ylabel('Historgamm')
ax2.set_xlabel('Classes')

td = pd.DataFrame({"index":error[:,0], "entropy":error[:,1] })

td.to_excel("./out/error.xls", sheet_name = "error")
td.to_csv("./out/error.txt", sep = " ")
plt.savefig(f"./out/plot_error.png")
plt.show()
