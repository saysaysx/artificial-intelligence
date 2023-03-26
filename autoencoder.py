# MIT License
# Copyright (c) 2023 saysaysx

import matplotlib.pyplot as plt
import tensorflow.keras as krs
import numpy
from keras.datasets import mnist
nw = 28
nh = 28
num_hide = 98
# загружаем примеры обучения mnist (рукописные цифры)
(trainx, trainy), (testx, testy) = mnist.load_data()
# нормируем от -1 до 1 изображения цифр
all_image = (trainx/255.0-0.5)*1.999
# добавляем дополнительное измерение соответствующее одной цветовой карте
all_image = numpy.expand_dims(all_image, axis=3)
# задаем входной слой экодера высота на ширину на количество карт
encoder_input = krs.layers.Input(shape=(nw,nh,1))
# задаем сверточный слой с 32 фильтрами-картами и фильтрами 3 на 3
# оставляет тот же размер карты 28*28
lay = krs.layers.Conv2D(32, (3, 3), strides = (2,2), activation='relu', padding='same')(encoder_input)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(64, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
# добавляем слой прореживания
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(128, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(256, (3, 3), strides = (2,2),activation='relu', padding='same')(lay)
# слой который многомерный тензорный слой превращает в плоский вектор
lay = krs.layers.Flatten()(lay)
# выходной кодирующий слой
lay_out_encoder = krs.layers.Dense(num_hide, activation="linear", name='den4')(lay)
# создаем сеть энкодера
encoder = krs.Model(encoder_input, lay_out_encoder)

# создание сети декодера, входной слой
decoder_input = krs.layers.Input(shape=(num_hide,))
lay = krs.layers.Dense(128*7*7)(decoder_input)
# преобразуем плоский слой в многомерный тензор 7*7*128
lay = krs.layers.Reshape(target_shape=(7,7,128))(lay)
lay = krs.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(lay)
# повышаем размерность карты в два раза, будет 14*14
# можно использовать билинейную интерполяцию если хотите
lay = krs.layers.UpSampling2D(size=(2,2))(lay)
lay = krs.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(lay)
lay = krs.layers.UpSampling2D(size=(2,2))(lay)
lay_out_decoder = krs.layers.Conv2D(1, (3, 3), activation='tanh', padding='same')(lay)
# создаем сеть декодера
decoder = krs.Model(decoder_input,lay_out_decoder)

# объединяем обе сети в автоэнкодер
lay_out = decoder(lay_out_encoder)
autoencoder = krs.Model(encoder_input,lay_out)
# полученный вид модели сохраняем в файле в виде изображения
krs.utils.plot_model(autoencoder, to_file='.\out\autoencoder.png', show_shapes=True)
# компилируем модель автоэнкодера с функцией потерь mse и скоростью обучения 0.0002
autoencoder.compile(loss='mean_squared_error', optimizer=krs.optimizers.Adam(learning_rate=0.0002),
                  metrics=['accuracy'])
# запускаем 40 эпох обучения с размером батча 4000
ep = 40
autoencoder.fit(x = all_image,y = all_image,batch_size = 4000,epochs = ep)

# получаем выход автоэнкодера, изображения который он получает
index = numpy.random.randint(0,len(all_image),9)
out_img = autoencoder.predict(all_image[index])
# выводим их на графике
fig = plt.figure(figsize=(5,5))
for i in range(3):
    for j in range(3):
        ax = fig.add_subplot(3,3,i*3+j+1)
        ax.imshow(out_img[i*3+j][:,:,0])
plt.show()
# реализуем работу  с энкодером получая скрытый кодовый слой
from scipy.cluster.vq import kmeans2
out_vec = encoder.predict(all_image)
# получим центроиды кластеров для 10 кластеров
centroid, label = kmeans2(out_vec, 10, minit='++')

# получим центроиды кластеров для 2 кластеров
centroid1, label1 = kmeans2(out_vec, 2, minit='++')

# считаем координаты кластера как разность с центроидом
out_vec1 = (out_vec - centroid1[0])**2
out_vec2 = (out_vec - centroid1[1])**2
# берем среднее значение
outm = out_vec1.mean(axis=1)
outstd = out_vec2.mean(axis=1)

coutm = centroid.mean(axis=1)
coutstd = centroid.mean(axis=1)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
# рисуем на графике кластер объектов в виде среднее дисперсия
for i in range(10):
    mask = label == i
    ax.scatter(outm[mask],outstd[mask])
plt.show()

# реализуем другой способ кластеризации, с помощью автоэкодера с дескриптором размером 2
# эти два значения и будут использоваться как двумерные координаты кластера
encoder_input1 = krs.layers.Input(shape=(num_hide))
lay = krs.layers.Dense(2000, activation="relu")(encoder_input1)
lay = krs.layers.Dense(500, activation="relu")(lay)
lay = krs.layers.Dense(100, activation="relu")(lay)
lay_out_encoder1 = krs.layers.Dense(2, activation="linear", name='den')(lay)
encoder1 = krs.Model(encoder_input1, lay_out_encoder1)
decoder_input1 = krs.layers.Input(shape=(2,))
lay = krs.layers.Dense(100, activation="relu")(decoder_input1)
lay = krs.layers.Dense(500, activation="relu")(lay)
lay = krs.layers.Dense(2000, activation="relu")(lay)
lay_out_decoder1 = krs.layers.Dense(num_hide, activation="linear")(lay)
decoder1 = krs.Model(decoder_input1,lay_out_decoder1)
lay_out1 = decoder1(lay_out_encoder1)
autoencoder1 = krs.Model(encoder_input1,lay_out1)
krs.utils.plot_model(autoencoder1, to_file='.\out\autoencoder1.png', show_shapes=True)
autoencoder1.compile(loss='mean_squared_error', optimizer=krs.optimizers.Adam(learning_rate=0.0002),
                  metrics=['accuracy'])

autoencoder1.fit(x = out_vec,y = out_vec,batch_size = 4000,epochs = ep*8)
out_vec = encoder1.predict(out_vec)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
centroid, label = kmeans2(out_vec, 10, minit='random')
print(label)
print(label.shape)

# рисуем полученные кластера цифр
for i in range(10):
  mask = label == i
  ax.scatter(out_vec[mask,0],out_vec[mask,1])
  plt.text(centroid[i,0], centroid[i,1], i , fontdict=None)
plt.show()
