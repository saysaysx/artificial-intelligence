# MIT License
# Copyright (c) 2023 saysaysx

import tensorflow as tf
import numpy

def tfpow(x,st):
    stf = tf.cast(st, dtype = tf.float32)
    zn = tf.where((st%2!=0) & (x<0.0),-1.0,1.0)

    return zn*tf.pow(tf.abs(x),stf)

class Layer:
    def __init__(self, n_inp, n_grid, n_pow):
        assert n_grid>=n_pow, "Степень полинома должна быть меньше количества узлов"

        self.n_inp = n_inp
        self.n_grid = n_grid
        self.n_pow = n_pow
        self.a = 0.0
        self.b = 1.0
        self.ab = tf.linspace(self.a, self.b, n_grid)
        self.y = tf.Variable(tf.random.normal(shape=(n_inp,n_grid))*0.01,  trainable=True)
        X = tf.repeat(self.ab[:,None], n_pow, axis = 1)
        self.st = tf.range(self.n_pow)#tf.linspace(0.0, self.n_pow-1, self.n_pow)
        st = tf.transpose(tf.repeat(self.st[:, None], n_grid, axis=1))
        X = tfpow(X,st)
        XT = tf.transpose(X)
        XTX = tf.matmul(XT,X)
        Xi = tf.linalg.inv(XTX)
        self.M = tf.matmul(Xi,XT)
        res = tf.matmul(self.M, tf.transpose(self.y))
    def ask(self,x):
        coefs = tf.transpose(tf.matmul(self.M, tf.transpose(self.y)))
        st  = self.st[:,None, None]
        X = tfpow(x,st)
        X = tf.transpose(X, [1,2,0])

        ans = tf.reduce_sum(X*coefs,axis=2)

        return ans

class Net:
    def __init__(self, n, n_grid, n_pow, optimizer):
        self.inner = [Layer(n,n_grid, n_pow) for i in range(2*n)]#
        self.outer = Layer(2*n, n_grid, n_pow)
        self.n_grid = n_grid
        self.n_pow  = n_pow
        self.optimizer = optimizer
        return
    @tf.function
    def learn(self,x,y):
        with tf.GradientTape(persistent=True) as tape:
            outs = []
            trainable = []
            for inner in self.inner:
                res = inner.ask(x)
                s = tf.reduce_sum(res, axis = -1)
                outs.append(s)
                trainable.append(inner.y)
            xo = tf.convert_to_tensor(outs)
            xo = tf.transpose(xo)
            tf.print(tf.shape(xo))
            res = self.outer.ask(xo)
            trainable.append(self.outer.y)
            yo = tf.reduce_sum(res, axis=-1)
            loss = tf.reduce_mean(tf.square(yo - y))


        grads = tape.gradient(loss, trainable)
        self.optimizer.apply_gradients(zip(grads, trainable))
        return loss
    def ask(self,x):
        outs = []
        trainable = []
        for inner in self.inner:
            res = inner.ask(x)
            s = tf.reduce_sum(res, axis=-1)
            outs.append(s)
            trainable.append(inner.y)
        xo = tf.convert_to_tensor(outs)
        xo = tf.transpose(xo)
        tf.print(tf.shape(xo))
        res = self.outer.ask(xo)
        trainable.append(self.outer.y)
        yo = tf.reduce_sum(res, axis=-1)
        return yo


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
net = Net(2,6,5, optimizer)
#x = tf.constant([[0.5,0.3],[0.6,0.4], [0.5,0.3]])
#y = x[:,0] + x[:,1]
N = 10000
ex_in = numpy.random.rand(N,2)
def gauss(x,xmean):
    return numpy.exp(-(x-xmean)**2)

#ex_out = ex_in[:,0] + ex_in[:,1]
ex_out = gauss(ex_in[:,0],ex_in[:,1])
for i in range(1500):
    indices = numpy.random.randint(0, N, size=200)
    x = tf.cast(ex_in[indices],tf.float32)
    y = tf.cast(ex_out[indices],tf.float32)
    loss = net.learn(x,y)
    print(f"loss {loss}")

n = 100
xx = numpy.linspace(0.0,1.0,n)
mx = numpy.linspace(0.5,0.5,n)
xn = numpy.array([xx,mx]).T
print(xn.shape)

yn = gauss(xx,mx)

xn = tf.cast(xn,tf.float32)
yy = net.ask(xn)

import matplotlib.pyplot as plt
plt.plot(xx,yy)
plt.plot(xx,yn)

plt.show()
#lay = Layer(2,7,5)
#x = lay.ask(tf.constant([[0.5,0.3],[0.6,0.4], [0.5,0.3]]))
#tf.print(x)

#y = tfpow(tf.constant([-2.0,-3.0,4.0]),tf.constant([0,1,2]))
#tf.print(y)
