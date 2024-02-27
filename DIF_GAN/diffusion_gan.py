import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Conv2DTranspose, Reshape,Embedding,ZeroPadding2D,Cropping2D, UpSampling2D, Dense, Input, concatenate, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, LeakyReLU
import numpy
numpy.set_printoptions(precision=4)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()


def model_unet(io_shape):

    def blockconv(size,filt, lay):
        lay = Conv2D(filters = filt, kernel_size=(size,size),padding='same') (lay)
        lay = BatchNormalization() (lay)
        lay = LeakyReLU(alpha=0.1) (lay)
        lay = Conv2D(filters = filt, kernel_size=(size,size),padding='same') (lay)
        lay = BatchNormalization() (lay)
        block = LeakyReLU(alpha=0.1) (lay)
        out = MaxPooling2D() (lay)
        return block, out
    def blockdconv(size, filt, lay, cblock):
        #lay = UpSampling2D(size=(2,2),interpolation="bilinear") (lay)
        lay = Conv2DTranspose(filters=filt, kernel_size=(size,size), strides=(2, 2), padding='same') (lay)
        lay = BatchNormalization() (lay)
        lay = LeakyReLU(alpha=0.1) (lay)
        lay = concatenate([lay, cblock])
        lay = Conv2D(filters = filt, kernel_size=(size,size),padding='same') (lay)
        lay = BatchNormalization() (lay)
        lay = LeakyReLU(alpha=0.1) (lay)
        lay = Conv2D(filters = filt, kernel_size=(size,size),padding='same') (lay)
        lay = BatchNormalization() (lay)
        lay = LeakyReLU(alpha=0.1) (lay)
        return lay
    inp = Input(shape=io_shape)
    inpt = Input(shape=(1,))
    emb = Embedding(1,io_shape[0]*io_shape[1]) (inpt)
    emb = Reshape((io_shape[0],io_shape[1],1)) (emb)
    lay = concatenate([inp,emb])

    shape = numpy.array(io_shape)
    print(shape)
    size = 2**(numpy.trunc(numpy.log2(shape[0:2]-0.01))+1.0)
    print(size)
    difsize = size - shape[0:2]
    difsize = difsize.astype(int)
    padx = difsize[0]//2, difsize[0]//2 + difsize[0] % 2
    pady = difsize[1]//2, difsize[1]//2 + difsize[1] % 2
    print(f"pad {padx}, {pady}")

    if difsize.sum()>0:
        lay = ZeroPadding2D(padding=(padx,pady)) (lay)

    block1, out1 = blockconv(3,6,lay)
    block2, out1 = blockconv(3,12,out1)
    block3, out1 = blockconv(3,24,out1)
    block4, out1 = blockconv(3,48,out1)

    upblock4 = blockdconv(3,48,out1,block4)
    upblock3 = blockdconv(3,24,upblock4,block3)
    upblock2 = blockdconv(3,12,upblock3,block2)
    upblock4 = blockdconv(3,6,upblock2,block1)
    out = Conv2D(filters = io_shape[2],kernel_size=(3,3), padding='same') (upblock4)

    if difsize.sum()>0:
        out = Cropping2D(cropping=(padx,pady)) (out)
    model = keras.Model(inputs=[inp, inpt], outputs=out)

    return model

nw = 28
nh = 28

from keras.datasets import mnist
(trainx, trainy), (testx, testy) = mnist.load_data()

all_image = (trainx/255.0-0.5)*2.0
N_ALL = len(all_image)
model = model_unet([nw,nh,1])
keras.utils.plot_model(model, to_file='./out/unet.png', show_shapes=True)

nt = 1000
itx = tf.range(nt,dtype=tf.int32)
beta = tf.linspace(1e-4,0.005,nt)
alpha = 1 - beta
alphprod = tf.math.cumprod(alpha)
tf.print(alphprod)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

def get_qxt(x,t):
    x = tf.cast(x,tf.float32)
    shape = tf.shape(x)
    noise = tf.random.normal(shape)
    xt = x*(alphprod[t]**0.5)+noise*((1-alphprod[t])**0.5)
    return xt



@tf.function
def learn_dif(x, n_ex, unet):
    with tf.GradientTape() as tape:
        x = tf.cast(x,tf.float32)
        ind = tf.random.uniform(shape=[n_ex],minval = 0,maxval = nt, dtype = tf.int32)
        bitx = tf.gather(itx, ind)
        balpha = tf.gather(alphprod, ind)[:,None,None]
        shape = tf.shape(x)
        noise = tf.random.normal(shape)
        xt = x*balpha**0.5+noise*(1-balpha)**0.5
        yt = unet([xt,bitx])
        lossp = tf.reduce_mean(tf.square(yt[:,:,:,0] - noise))

    trainable_vars = unet.trainable_variables

    grads = tape.gradient(lossp, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    return xt,yt, lossp


def resolve(noise, unet, stsave = 200):
    val = []
    xt = tf.cast(noise,tf.float32)
    for i in range(nt-1,0,-1):
        shape = tf.shape(xt)
        rand = tf.random.normal(shape)
        err = unet([xt,tf.reshape(itx[i],shape=(1,1))])
        sig = tf.sqrt((1 - alphprod[i-1])*beta[i]/(1 - alphprod[i]))
        salpha = 1/tf.sqrt(alpha[i])
        coefal = (1-alpha[i])/tf.sqrt(1-alphprod[i])
        xt = salpha * (xt-coefal*err)+sig*rand
        if(i%stsave==0):
            dif =  tf.reduce_max(xt[0]) - tf.reduce_min(xt[0])
            val.append((xt[0]-tf.reduce_min(xt[0]))/dif)


    err = unet([xt,tf.reshape(itx[0],shape=(1,1))])
    salpha = 1/tf.sqrt(alpha[0])
    coefal = (1-alpha[0])/tf.sqrt(1-alphprod[0])
    xt = salpha * (xt-coefal*err)
    dif = tf.reduce_max(xt[0]) - tf.reduce_min(xt[0])
    return (xt[0]-tf.reduce_min(xt[0]))/dif  , val


import time

N_BATCH = 1000
N_EPOCHS = 15000
try:
    model = keras.models.load_model("./out/model.h5")
    print("Load model ok")
except Exception as error:
    print(f"Load error {error}")
    pass

for i in range(N_EPOCHS):
    indicies  = numpy.random.randint(0,N_ALL,N_BATCH)
    x_ind = all_image[indicies]
    xt, yt, lossp = learn_dif(x_ind, N_BATCH, model)
    print(f"epoch {i} loss {lossp}")
    time.sleep(1.0)
model.save("./out/model.h5")

def norm(x):
    v = (x-x.min())/(x.max()-x.min())
    return v


n = 5
image = []
for i in range(n):
    num = int(nt*i/n)
    xt = get_qxt(numpy.array([all_image[0]]), num)
    #res = model.predict([xt,numpy.array([num])])
    image.append(norm(xt[0].numpy()))

image = numpy.concatenate(image,axis=1)

npr = 3
image1 = []
image2 = []



for i in range(npr):

    noise = numpy.random.randn(1,nh,nw,1)
    print(noise.shape)
    xt, val = resolve(noise,model)
    image1.append(xt)
    image2.append(numpy.concatenate(val,axis=1))


image1 = numpy.concatenate(image1,axis=1)
import matplotlib.pyplot as plot
plot.subplot(3, 1, 1)
plot.imshow(norm(image))
plot.subplot(3, 1, 2)
plot.imshow(norm(image1))
plot.subplot(3, 1, 3)
plot.imshow(norm(image2[0]))
print(image1.max())
print(image1.min())
print(image2[0].max())
print(image2[0].min())


plot.show()
