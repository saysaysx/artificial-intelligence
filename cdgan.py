
# MIT License
# Copyright (c) 2023 saysaysx

# Conditional GAN

import matplotlib.pyplot as plt
import tensorflow.keras as krs
import numpy
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist

# функция устанавливающая обучаемость слоев модели, False - веса фиксированные

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#gpus = tf.config.list_physical_devices('GPU')
#print("GPUs Available: ", gpus)

#tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

nw = 28
nh = 28
num_hide = 196

# загружаем обучающий датасет
(trainx, trainy), (testx, testy) = mnist.load_data()

all_image = (trainx/255.0-0.5)*1.999
all_image = numpy.expand_dims(all_image, axis=3)

# преобразуем метки классов из вида 0..9 в бинарный вектор
all_labels = krs.utils.to_categorical(trainy)

print(all_image.shape)

# создаем сеть дескриминатор с двумя входами
desc_input = krs.layers.Input(shape=(nw,nh,1),name = 'inpd1')
# второй вход для метки класса образа
desc_input1 = krs.layers.Input(shape=(10,), name = 'inpd2')
lay1 = krs.layers.Dense(7*7, name = 'd2') (desc_input1)
lay1 = krs.layers.Dense(28*28, name = 'd3') (lay1)
lay1 = krs.layers.Reshape((28,28,1), name = 'd4') (lay1)
# объединяем два параллельных слоя
layc = krs.layers.Concatenate(name = 'd5') ([desc_input, lay1])


lay = krs.layers.Conv2D(32, (3, 3), strides = (2,2), activation='relu', padding='same', name = 'd6')(layc)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(64, (3, 3), strides = (2,2),activation='relu', padding='same', name = 'd7')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(128, (3, 3), strides = (2,2),activation='relu', padding='same', name = 'd8')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(256, (3, 3), strides = (2,2),activation='relu', padding='same', name = 'd9')(lay)
lay = krs.layers.Flatten(name = 'd10')(lay)
lay_out = krs.layers.Dense(1, activation="sigmoid", name='den4')(lay)

# определяем модель дескриминатора
descriminator = krs.Model([desc_input,desc_input1], lay_out)

# создаем сеть генератора с двумя входами
gen_input = krs.layers.Input(shape=(num_hide,),name = 'ginp1')
# вход для метки класса образа
gen_input1 = krs.layers.Input(shape=(10,),name = 'ginp2')

layc = krs.layers.Concatenate(name = 'g3')([gen_input,gen_input1])
lay = krs.layers.Dense(128*7*7,name = 'g4')(layc)
lay = krs.layers.Reshape(target_shape=(7,7,128),name = 'g5')(lay)
lay = krs.layers.Conv2D(128, (3, 3), activation='relu', padding='same',name = 'g6')(lay)
lay = krs.layers.UpSampling2D(size=(2,2),name = 'g7')(lay)
lay = krs.layers.Conv2D(64, (3, 3), activation='relu', padding='same',name = 'g8')(lay)
lay = krs.layers.UpSampling2D(size=(2,2),name = 'g9')(lay)
lay_out = krs.layers.Conv2D(1, (3, 3), activation='tanh', padding='same',name = 'g10')(lay)
# создание модели генератора
generator = krs.Model([gen_input,gen_input1],lay_out)

# сохраняем полученные модели
krs.utils.plot_model(descriminator, to_file='./out/descriminator.png', show_shapes=True)
krs.utils.plot_model(generator, to_file='./out/generator.png', show_shapes=True)

# создаем объединенную сеть дескриминатора и генератора

gan = descriminator([generator.layers[-1].output, gen_input1])
gan_model = krs.Model(inputs=[gen_input, gen_input1], outputs=gan)
# binary_crossentropy

optimizerd = krs.optimizers.Adam(learning_rate=0.0001,clipnorm=1.0)
optimizerg = krs.optimizers.Adam(learning_rate=0.0001,clipnorm=1.0)

krs.utils.plot_model(gan_model, to_file='./out/gan_model.png', show_shapes=True)

n_learn = 10000
n_batch = 32
n_batch_check = 2000
ones = numpy.ones((n_batch, 1))   -1e-13
zeros = numpy.zeros((n_batch, 1))   +1e-13
oz = numpy.concatenate([ones, zeros])
zo = numpy.concatenate([zeros, ones])

ones_c = numpy.ones((n_batch_check, 1))   -1e-13
zeros_c = numpy.zeros((n_batch_check, 1))   +1e-13
oz_c = numpy.concatenate([ones_c, zeros_c])
zo_c = numpy.concatenate([zeros_c, ones_c])


# Рисуем примеры образов из обучающей выборки
fig = plt.figure(figsize = (5,5))
bx = [[]]*16
for i in range(16):
    bx[i] = fig.add_subplot(4, 4, i + 1)
    bx[i].imshow((all_image[i][:, :, 0]+1)*0.5)




ax = [[], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []]
# загружаем модель классификатора
classificator = krs.models.load_model("./out/class_mnist.h5")

error = []

print("Start learning!!!")

values = numpy.arange(all_image.shape[0])
values_cl = numpy.arange(10)

@tf.function
def train_descriminator(rf,labels,oz):
    with tf.GradientTape() as tape:
        d = descriminator([rf,labels],training = True)
        loss = - tf.reduce_mean(oz*tf.math.log(d+1.0e-12)+(1.0-oz)*tf.math.log(1.0-d+1.0e-12))
        trainable_vars = descriminator.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizerd.apply_gradients(zip(grads, trainable_vars))
    return loss

@tf.function
def train_gan(rx,fake_labels):
    with tf.GradientTape() as tape:
        d = gan_model([rx,fake_labels],training = True)
        loss = - tf.reduce_mean(tf.math.log(d+1e-12))

        trainable_vars = generator.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizerg.apply_gradients(zip(grads, trainable_vars))
    return loss





for i_learn in range(n_learn):
    # выбираем батч реальных образов и меток их классов

    indexes = numpy.random.choice(values, size=n_batch, replace=False)
    #indexes = numpy.random.randint(0, all_image.shape[0], n_batch)
    real_image = all_image[indexes]
    real_labels = all_labels[indexes]

    # генерируем вектора скрытого пространства и метки классов
    rx = numpy.random.randn(n_batch, num_hide)

    fake_labels = numpy.random.randint(0, 10, n_batch)
    fake_labels = krs.utils.to_categorical(fake_labels, num_classes = 10)

    # получаем генерируемые образы
    fake_image = generator.predict([rx,fake_labels], batch_size=n_batch, verbose = 0)

    # склеиваем в один батч реальные и фейковые образы
    rf = numpy.concatenate([real_image,fake_image])
    oz = numpy.concatenate([ones,zeros])

    # склеиваем метки реальных и фейковых образов
    labels = numpy.concatenate([real_labels,fake_labels])
    # обучаем дескриминатор


    rf = tf.cast(rf,tf.float32)
    labels = tf.cast(labels,tf.float32)
    oz = tf.cast(oz,tf.float32)
    dloss = train_descriminator(rf,labels,oz)


    # обучаем генератор
    #gloss1 = gan_model.train_on_batch([rx,fake_labels], ones)
    rx = tf.cast(rx,tf.float32)
    fake_labels = tf.cast(fake_labels,tf.float32)
    gloss1 = train_gan(rx,fake_labels)


    zo = numpy.concatenate([zeros,ones])

    # контроль обучения, сохранение результатов
    if (i_learn % 1000 == 0 ):
        indexes = numpy.random.choice(values, size=n_batch_check, replace=False)
        #indexes = numpy.random.randint(0, all_image.shape[0], n_batch_check)
        real_image = all_image[indexes]
        real_labels = all_labels[indexes]

        rx = numpy.random.normal(0, 1, (n_batch_check, num_hide))#numpy.random.randn(n_batch, num_hide)
        inp_dec =  rx

        fake_labels = numpy.random.randint(0, 10, n_batch_check)
        fake_labels = krs.utils.to_categorical(fake_labels, num_classes = 10)


        fake_image = generator.predict([inp_dec, fake_labels], verbose=0)
        images = numpy.concatenate([real_image, fake_image])

        labels = numpy.concatenate([real_labels,fake_labels])
        pred = descriminator.predict_on_batch([images, labels])
        dloss = numpy.abs(pred - oz_c).mean()

        # для расчета gloss
        ans_gan = gan_model.predict_on_batch([inp_dec,fake_labels])

        # получаем метки сгенерированных образов с помощью предобученного классификатора
        ans_cls = classificator.predict_on_batch(fake_image)

        # получаем гистограмму распределения классов
        arr = numpy.bincount(ans_cls.argmax(axis=1))
        print(arr)
        # получаем вероятность появления класса
        parr = arr/arr.sum()+1e-20
        # считаем энтропию
        H = (parr*numpy.log(1/parr)).sum()
        error.append([i_learn, H , dloss])
        print(f"entropy {H}")

        gloss = numpy.abs(ans_gan - ones_c).mean()

        print(f"index {i_learn} dloss {dloss}  gloss {gloss} ")

        if (i_learn % 2000 == 0 and i_learn > 4000):
            # сохраняем веса дескриминатора и генератора
            descriminator.save_weights("./out/model_descriminator1.h5")
            generator.save_weights("./out/model_generator1.h5")

        # рисуем 16 примеров сгенерированных изображений

        rx = numpy.random.normal(0, 1, (16, num_hide))#numpy.random.randn(16,num_hide)

        fake_labels = numpy.random.randint(0, 10, 16)
        fake_labels = krs.utils.to_categorical(fake_labels, num_classes = 10)


        inp_dec = rx
        pred_dec = generator.predict([inp_dec, fake_labels], verbose=0)

        if(i_learn%2000 ==0):
            fig = plt.figure(figsize = (5,5))
            for i in range(16):
                ax[i] = fig.add_subplot(4, 4, i + 1)
                ax[i].imshow((pred_dec[i][:, :, 0]+1)*0.5)


            plt.show()

# отображаем поведение энтропии в процессе обучения
error  = numpy.array(error)
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(error[:,0],error[:,1])
ax1.set_ylabel('Entropy')
ax1.set_xlabel('Number of iteration')


rx = numpy.random.randn(n_batch_check,num_hide)

fake_labels = numpy.random.randint(0, 10, n_batch_check)
fake_labels = krs.utils.to_categorical(fake_labels, num_classes = 10)


fake_image = generator.predict([rx,fake_labels])

ans_cls = classificator.predict_on_batch(fake_image)
arr = numpy.bincount(ans_cls.argmax(axis=1))


ax2.bar([0,1,2,3,4,5,6,7,8,9], arr)
ax2.set_ylabel('Historgamm')
ax2.set_xlabel('Classes')

td = pd.DataFrame({"index":error[:,0], "entropy":error[:,1], "dloss":error[:,2] })

td.to_excel("./out/error.xls", sheet_name = "error")
td.to_csv("./out/error.txt", sep = " ")
plt.show()
