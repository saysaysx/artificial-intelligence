# MIT License
# Copyright (c) 2025 saysaysx
# Conditional GAN

import matplotlib.pyplot as plt
import tensorflow.keras as krs
import numpy
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from tensorflow.keras import layers
import io
import zipfile
from PIL import Image


def process_zip_archive(zip_path, target_dir='train/'):
    class_images = {}  # Словарь для хранения {класс: [изображения]}

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Фильтруем файлы по целевой директории
        file_list = [f for f in zip_ref.namelist()
                    if f.startswith(target_dir) and not f.endswith('/')]

        for file_path in file_list:
            # Определяем класс из структуры директорий
            # Путь вида: 'train/class1/img1.jpg'
            parts = file_path.split('/')
            if len(parts) > 2:  # Есть вложенность директорий
                class_name = parts[1]  # Берем имя директории как класс

                # Читаем изображение
                with zip_ref.open(file_path) as file:
                    try:
                        img = Image.open(io.BytesIO(file.read()))

                        # Добавляем в словарь классов
                        if class_name not in class_images:
                            class_images[class_name] = []
                        class_images[class_name].append(img)

                        #print(f"Обработано: {file_path} | Класс: {class_name}")
                    except Exception as e:
                        print(f"Ошибка чтения {file_path}: {e}")

    return class_images

def prepare_dataset(images_dict, target_size=(64, 64)):
    # Создаем списки для данных и меток
    x = []
    y = []

    # Создаем словарь для соответствия имени класса и числовой метки
    class_names = sorted(list(images_dict.keys()))
    class_to_label = {class_name: i for i, class_name in enumerate(class_names)}

    # Обрабатываем каждый класс
    for class_name, img_list in images_dict.items():
        for img in img_list:
            try:
                # Преобразуем изображение
                img = img.convert('RGB')  # Конвертируем в RGB
                img = img.resize(target_size)  # Изменяем размер
                img_array = numpy.array(img)  # Преобразуем в numpy массив

                # Нормализуем значения пикселей [0, 255] -> [0, 1]
                img_array = img_array / 255.0

                x.append(img_array)
                y.append(class_to_label[class_name])
            except Exception as e:
                print(f"Ошибка обработки изображения класса {class_name}: {e}")

    # Преобразуем списки в numpy массивы
    x = numpy.array(x)
    y = numpy.array(y)

    return x, y, class_to_label

# Использование
images = process_zip_archive(zip_path='Linnaeus.zip', target_dir='train/')
x, y, class_mapping = prepare_dataset(images)

print(f"Форма массива изображений: {x.shape}")  # (n_samples, 64, 64, 3)
print(f"Форма массива меток: {y.shape}")       # (n_samples,)
print(f"Соответствие классов и меток: {class_mapping}")


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)


nw = 64
nh = 64
num_hide = 128
nclasses = len(class_mapping)


all_image = (x-0.5)*1.999

# преобразуем метки классов из вида 0..9 в бинарный вектор
all_labels = krs.utils.to_categorical(y)

print(all_image.shape)


augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.01),
        layers.RandomZoom((-0.15, 0.15)),

    ])


# создаем сеть дескриминатор с двумя входами
desc_input = krs.layers.Input(shape=(nw,nh,3),name = 'inpd1')
# второй вход для метки класса образа
desc_input1 = krs.layers.Input(shape=(nclasses,), name = 'inpd2')
lay1 = krs.layers.Dense(16*16, name = 'd2') (desc_input1)
lay1 = krs.layers.Reshape((16,16,1), name = 'd4') (lay1)
lay1 = krs.layers.UpSampling2D(size=(4, 4), interpolation='nearest', name = 'd44') (lay1)
# объединяем два параллельных слоя
layc = krs.layers.Concatenate(name = 'd5') ([desc_input, lay1])


lay = krs.layers.Conv2D(64, (5, 5), strides = (2,2), padding='same', name = 'd6')(layc)
lay = layers.LeakyReLU(0.2)(lay)
lay = krs.layers.Conv2D(128, (5, 5), strides = (2,2), padding='same', name = 'd7')(lay)
lay = layers.BatchNormalization()(lay)
lay = layers.LeakyReLU(0.2)(lay)
lay = krs.layers.Conv2D(256, (5, 5), strides = (2,2), padding='same', name = 'd8')(lay)
lay = layers.BatchNormalization()(lay)
lay = layers.LeakyReLU(0.2)(lay)
lay = krs.layers.Conv2D(512, (5, 5), strides = (2,2), padding='same', name = 'd9')(lay)
lay = layers.BatchNormalization()(lay)
lay = layers.LeakyReLU(0.2)(lay)
lay = krs.layers.Flatten(name = 'd10')(lay)
lay_out = krs.layers.Dense(1, activation="sigmoid", name='den4')(lay)

# определяем модель дескриминатора
descriminator = krs.Model([desc_input,desc_input1], lay_out)

# создаем сеть генератора с двумя входами
gen_input = krs.layers.Input(shape=(num_hide,),name = 'ginp1')
# вход для метки класса образа
gen_input1 = krs.layers.Input(shape=(nclasses,),name = 'ginp2')

layc = krs.layers.Concatenate(name = 'g3')([gen_input,gen_input1])
lay = krs.layers.Dense(512*4*4,name = 'g4')(layc)
lay = krs.layers.Reshape(target_shape=(4,4,512),name = 'g5')(lay)
lay = krs.layers.Conv2D(512, (5, 5),  padding='same', use_bias=False)(lay)
lay = layers.BatchNormalization()(lay)
lay = layers.LeakyReLU(0.2)(lay)
lay = krs.layers.UpSampling2D(size=(4,4),name = 'g7', interpolation='bilinear')(lay)
lay = krs.layers.Conv2D(256, (5, 5),  padding='same', use_bias=False)(lay)
lay = layers.BatchNormalization()(lay)
lay = layers.LeakyReLU(0.2)(lay)
lay = krs.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(lay)
lay = krs.layers.Conv2D(128, (5, 5),  padding='same', use_bias=False)(lay)
lay = layers.BatchNormalization()(lay)
lay = layers.LeakyReLU(0.2)(lay)
lay = krs.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(lay)

lay_out = krs.layers.Conv2D(3, (7, 7), activation='tanh', padding='same',name = 'g12')(lay)
# создание модели генератора
generator = krs.Model([gen_input,gen_input1],lay_out)

# сохраняем полученные модели
krs.utils.plot_model(descriminator, to_file='./out/descriminator.png', show_shapes=True)
krs.utils.plot_model(generator, to_file='./out/generator.png', show_shapes=True)

# создаем объединенную сеть дескриминатора и генератора

gan = descriminator([generator.layers[-1].output, gen_input1])
gan_model = krs.Model(inputs=[gen_input, gen_input1], outputs=gan)
# binary_crossentropy

optimizerd = krs.optimizers.Adam(learning_rate=0.00003)
optimizerg = krs.optimizers.Adam(learning_rate=0.00003)

krs.utils.plot_model(gan_model, to_file='./out/gan_model.png', show_shapes=True)

n_learn = 800000
n_batch = 32
n_batch_check = 400
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
    bx[i].imshow((all_image[i][:, :]+1)*0.5)




ax = [[], [], [], [], [], [], [], [],[], [], [], [], [], [], [], []]
# загружаем модель классификатора
classificator = krs.models.load_model("./classifier.h5")

error = []

print("Start learning!!!")

values = numpy.arange(all_image.shape[0])
values_cl = numpy.arange(10)

@tf.function
def train_descriminator(real_img,labels):
    with tf.GradientTape() as tape:
        real_aug = augmentation(real_img)
        real_out = descriminator([real_aug,labels],training = True)
        rx =  tf.random.normal(shape=(n_batch, num_hide))
        fake_img = generator([rx,labels])
        fake_out = descriminator([fake_img,labels],training = True)
        e = 1e-8
        fake_out = tf.clip_by_value(fake_out,0.0,1.0-e)
        real_out = tf.clip_by_value(real_out,e,1.0)

        #loss_critic = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
        loss_critic = tf.reduce_mean(-tf.math.log(real_out) - tf.math.log(1-fake_out))

        xval = tf.reduce_mean(tf.abs(real_img[:,None]-real_img[None,:]),axis=[-3,-2,-1])
        fval = tf.abs(real_out[:,None]-real_out[None,:])[:,:,0]
        penalty1 = tf.reduce_mean(tf.math.exp(fval-xval-5))

        xval = tf.reduce_mean(tf.abs(fake_img[:,None]-fake_img[None,:]),axis=[-3,-2,-1])
        fval = tf.abs(fake_out[:,None]-fake_out[None,:])[:,:,0]
        penalty2 = tf.reduce_mean(tf.math.exp(fval-xval-5))

        loss = loss_critic+penalty1+penalty2

        trainable_vars = descriminator.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizerd.apply_gradients(zip(grads, trainable_vars))
    return loss

@tf.function
def train_gan(fake_labels):
    with tf.GradientTape() as tape:
        rx =  tf.random.normal(shape=(n_batch, num_hide))
        img = generator([rx,fake_labels],training = True)
        d = descriminator([img,fake_labels], training = True)
        e = 1e-12
        d = tf.clip_by_value(d,e,1.0)

        xval = tf.reduce_mean(tf.abs(img[:,None]-img[None,:]),axis=[-3,-2,-1])
        fval = tf.abs(d[:,None]-d[None,:])[:,:,0]
        penalty = tf.reduce_mean(tf.math.exp(fval-xval-5))

        #losscl = tf.reduce_mean(tf.nn.relu(img - 1)+ tf.nn.relu(-img))*100.0
        #lp   =  - tf.reduce_mean(d)
        lp = tf.reduce_mean(-tf.math.log(d))

        loss = lp + penalty


        trainable_vars = generator.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    optimizerg.apply_gradients(zip(grads, trainable_vars))
    return loss


fig = plt.figure(figsize = (8,8))
for i in range(16):
    ax[i] = fig.add_subplot(4, 4, i + 1)



for i_learn in range(n_learn):
    # выбираем батч реальных образов и меток их классов
    indexes = numpy.random.choice(values, size=n_batch, replace=False)

    real_image = all_image[indexes]
    real_labels = all_labels[indexes]
    # генерируем вектора скрытого пространства и метки классов

    real_image = tf.cast(real_image,tf.float32)
    real_labels = tf.cast(real_labels,tf.float32)
    dloss = train_descriminator(real_image,real_labels)

    # обучаем генератор
    gloss1 = train_gan(real_labels)


    zo = numpy.concatenate([zeros,ones])

    # контроль обучения, сохранение результатов
    if (i_learn % 1000 == 0 ):
        indexes = numpy.random.choice(values, size=n_batch_check, replace=False)
        #indexes = numpy.random.randint(0, all_image.shape[0], n_batch_check)
        real_image = all_image[indexes]
        real_labels = all_labels[indexes]

        rx = numpy.random.normal(0, 1, (n_batch_check, num_hide))#numpy.random.randn(n_batch, num_hide)
        inp_dec =  rx

        fake_labels = numpy.random.randint(0, nclasses, n_batch_check)
        fake_labels = krs.utils.to_categorical(fake_labels, num_classes = nclasses)


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
            descriminator.save_weights("./out/model_descriminator1.weights.h5")
            generator.save_weights("./out/model_generator1.weights.h5")

        # рисуем 16 примеров сгенерированных изображений

        rx = numpy.random.normal(0, 1, (16, num_hide))#numpy.random.randn(16,num_hide)

        fake_labels = numpy.random.randint(0, nclasses, 16)
        fake_labels = krs.utils.to_categorical(fake_labels, num_classes = nclasses)


        inp_dec = rx
        pred_dec = generator.predict([inp_dec, fake_labels], verbose=0)

        if(i_learn%2000 ==0):
            for i in range(16):
                ax[i].imshow((pred_dec[i][:, :]+1)*0.5)
                plt.savefig("./out/carts.png")




# отображаем поведение энтропии в процессе обучения
error  = numpy.array(error)
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(error[:,0],error[:,1])
ax1.set_ylabel('Entropy')
ax1.set_xlabel('Number of iteration')


rx = numpy.random.randn(n_batch_check,num_hide)

fake_labels = numpy.random.randint(0, nclasses, n_batch_check)
fake_labels = krs.utils.to_categorical(fake_labels, num_classes = nclasses)


fake_image = generator.predict([rx,fake_labels])

ans_cls = classificator.predict_on_batch(fake_image)
arr = numpy.bincount(ans_cls.argmax(axis=1))


ax2.bar([i for i in range(nclasses)], arr)
ax2.set_ylabel('Historgamm')
ax2.set_xlabel('Classes')

td = pd.DataFrame({"index":error[:,0], "entropy":error[:,1], "dloss":error[:,2] })

td.to_excel("./out/error.xls", sheet_name = "error")
td.to_csv("./out/error.txt", sep = " ")
plt.show()
