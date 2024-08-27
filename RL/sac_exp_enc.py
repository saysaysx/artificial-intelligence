
from ipywidgets import interact

# MIT License
# Copyright (c) 2023 saysaysx
import random
import numpy
import time

from random import choices
import tensorflow as tf
#import gymnasium as gym
import matplotlib.pyplot as plt
import pandas


from ale_py import ALEInterface
ale = ALEInterface()
from ale_py.roms import Breakout
pong_rom = ale.loadROM(Breakout)

import  gym



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization, Dropout, Conv2D, Reshape, Flatten
from tensorflow.keras.layers import MaxPooling1D, Permute, Conv1D, LSTM, LeakyReLU, Cropping1D, Multiply, Softmax, GaussianNoise
from tensorflow.keras.layers import RepeatVector, Subtract, Embedding
import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

def get_ind_r(x):
    x = x+0.0012
    f = numpy.cumsum(x)/(1+0.0012*len(x))
    v = numpy.random.random()
    index = numpy.digitize(v,f)
    if index>=len(x):
        index = len(x)-1
    return index





numpy.set_printoptions(precision=4)

#print("Start gput opts")
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

#sess = tf.compat.v1.Session(config=config)
#sess.as_default()


def steps(n):
    index = 0
    def step():
        nonlocal index
        while True:
            index = index + 1

            if index>=n:
                index = 0
            if index%n==0:
                yield True
            else:
                yield False



    return step

class environment():
    def __init__(self):
        self.env = gym.make("Breakout-ram-v4", render_mode="rgb_array")
        #self.env = gym.make("MsPacman-ram-v4", render_mode="rgb_array")

        self.env = gym.wrappers.FrameStack(self.env,num_stack=4)
        print(self.env.action_space.n)
        self.n_action = self.env.action_space.n
        print("Максимальная награда за действие:", self.env.reward_range)

        self.step = 0

        self.reward = 0.0
        self.index = 0
        print(self.env.observation_space)
        shx = (self.env.observation_space.high).transpose([0,1]).shape
        print(shx)
        self.state_max = numpy.reshape(self.env.observation_space.high.transpose([0,1]),[shx[0],shx[1]])
        self.state_min = numpy.reshape(self.env.observation_space.low.transpose([0,1]),[shx[0],shx[1]])


        self.igame  = 0
        self.xnew = []
        self.ynew = []
        self.observ = [[],[],[],[],[]]
        self.ind_obs = 0
        self.n_obs = 1

        self.env_reset()
        self.state()




    def get_state(self, act):
        self.igame = self.igame + 1
        self.step = self.step + 1
        if(self.index == 0):
            self.reward  = 0.0
        self.ind_obs = 0


        next_observation, reward, done, _,_ = self.env.step(act)
        reward = reward
        next_observation = numpy.array(next_observation)


        shx = (next_observation).transpose([0,1]).shape
        self.field = numpy.reshape(next_observation.transpose([0,1]),[shx[0],shx[1]])


        self.reward = self.reward+reward
        self.index = self.index + 1


        return self.state(), reward, done

    def env_reset(self):
        self.index = 0
        self.reward = 0
        im = numpy.array(self.env.reset()[0])
        shx = im.transpose([0,1]).shape
        next_observation = numpy.reshape(im.transpose([0,1]),[shx[0],shx[1]])


        self.field = next_observation

    def get_image(self):
        x = self.env.render()

        return x

    def state(self):
        state = self.field

        return state#(( state - self.state_min) / (self.state_max - self.state_min))

    def get_shape_state(self):

        return self.state().shape
    def get_len_acts(self):
        return self.n_action

    def get_len_state(self):
        return len(self.state())


print("Start declare widgets")

import ipywidgets as widgets
from datetime import datetime

from ipywidgets import Image
from io import BytesIO
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image_widget1 = Image(value=buf.getvalue(), format='png')
#display(image_widget1)


text_widget = widgets.Label(value="This is some text to display.")
#display(text_widget)

def plotv(x,y,im):

    #plt.plot(x,y)
    fig,ax = plt.subplots(1,2,figsize=(10,2))
    ax[0].imshow(im)
    ax[1].plot(x,y)


    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plt.close()
    image_widget1.value = buf.getvalue()

print("Declare function plot")



class sac:
    def __init__(self,env):
        self.start_time = datetime.now()

        self.env = env
        self.index = 0

        self.max_size = self.index
        self.flag = False

        self.shape_state = env.get_shape_state()
        self.len_act = env.get_len_acts()
        self.gamma = 0.999
        self.alpha = tf.Variable(1.0)


        self.max_t = 10


        self.T = 512
        self.n_buffer = 80000
        self.buf_index  = 0
        self.flag_buf = False
        self.indT = tf.range(self.T)

        print("Make bufs")

        self.rews = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.acts = numpy.zeros((self.n_buffer, 1),dtype=numpy.float32)
        self.policies = numpy.zeros((self.n_buffer, self.len_act),dtype=numpy.float32)
        self.values = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.previous_states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.dones = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.vars = [i for i in range(self.len_act)]

        print("Start make networks")
        from keras.layers import Lambda
        from tensorflow.keras.utils import register_keras_serializable
        @register_keras_serializable()
        def rescale(x):
            y =  tf.cast(x, tf.float32)/255-0.5
            y = Reshape((512,)) (y)
            return y

        resc = Lambda(rescale)

        @register_keras_serializable()
        def maximumf(x):
            y = tf.reduce_max(x,axis=-1)
            return y
        maxf = Lambda(maximumf)

        @register_keras_serializable()
        def add_two(v):
            x, y = v
            z = x+y
            return z
        add_l = Lambda(add_two)

        @register_keras_serializable()
        def critic_out(v):
            v = tf.nn.softplus(v-1)
            return v
        critic_out_lay = Lambda(critic_out)

        print("Shape")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state,  dtype='uint8')
        lay = resc(inp1)
        lay = Dense(450, activation = 'relu') (lay)
        lay = Dense(350, activation = 'relu') (lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = Dense(100, activation = 'relu') (lay)

        layv1 = Dense(self.len_act, activation = 'linear') (lay)



        self.nnets = 9
        self.modelq = [[]]*self.nnets
        self.modelq[0] = keras.Model(inputs=inp1, outputs=[layv1])

        tf.keras.utils.plot_model(self.modelq[0], to_file='./out/netq.png', show_shapes=True)

        for i in range(1,self.nnets):
            self.modelq[i] = keras.models.clone_model(self.modelq[0])



        print("------------")
        print(self.shape_state)
        self.step_s = steps(4)

        @register_keras_serializable()
        def scale(x):
            qe = tf.nn.softmax(x)
            return qe
        sc = Lambda(scale)

        @register_keras_serializable()
        def norm(x):
            y1 = tf.cos(x)**2
            y2 = tf.sin(x)**2
            y = tf.concat([y1,y2],axis=1)

            return y
        norms = Lambda(norm)



        @register_keras_serializable()
        def sum(x):
            y = tf.stack(x)
            y  = tf.reduce_mean(y,axis=0)
            return y
        suml = Lambda(sum)


        inp1 = Input(shape = self.shape_state, dtype='uint8' )

        lay = resc(inp1)
        coefv = 1e-3*self.len_act

        lay = Dense(450, activation = 'relu') (lay)
        lay = Dense(350, activation = 'relu') (lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = Dense(100, activation = 'relu') (lay)
        layp1 = Dense(self.len_act, activation="linear") (lay)
        layp = sc(layp1)


        self.modelp = keras.Model(inputs=inp1, outputs=layp)

        self.modelpi = keras.models.clone_model(self.modelp)
        self.modelp_max = keras.models.clone_model(self.modelp)


        inp1 = Input(shape = self.shape_state,  dtype='uint8')
        lay = resc(inp1)
        inp2 = Input(shape = (1,))
        lay2 = Dense(50, activation = 'relu') (inp2)
        lay = Dense(350, activation = 'relu') (lay)
        lay = concatenate([lay,lay2])
        lay = BatchNormalization()(lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = BatchNormalization()(lay)
        lay = Dense(150, activation = 'relu') (lay)
        lay = BatchNormalization()(lay)
        lay = Dense(80, activation = 'relu') (lay)
        lay = BatchNormalization()(lay)
        lay = Dense(150, activation = 'relu') (lay)
        lay = BatchNormalization()(lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = BatchNormalization()(lay)
        lay = Dense(350, activation = 'relu') (lay)
        lay = BatchNormalization()(lay)
        lay = Dense(512, activation = 'linear') (lay)
        layo = Reshape(self.shape_state) (lay)
        self.modelaenc = keras.Model(inputs=[inp1,inp2], outputs=layo)




        self.targetaq = keras.models.clone_model(self.modelq[0])
        self.modelaq = keras.models.clone_model(self.modelq[0])

        self.targetq = [keras.models.clone_model(self.modelq[i]) for i in range(self.nnets)]

        self.nrewards  = 3
        self.max_rewards = numpy.array([0.0]*self.nrewards)


        tf.keras.utils.plot_model(self.targetq[0], to_file='./out/nettq1.png', show_shapes=True)
        tf.keras.utils.plot_model(self.modelp, to_file='./out/netp.png', show_shapes=True)


        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.optimizera = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.optimizeraq = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.optimizerap = tf.keras.optimizers.Adam(learning_rate=0.0002)

        self.alphav = tf.Variable(0.005)
        self.alpha_val = tf.Variable(0.0)
        self.alphat = tf.constant([0.009,0.009,0.009,0.009,0.009])
        self.aut_loss = tf.Variable(0.000)

        self.border = tf.Variable(0.02)
        self.std_rnd = tf.Variable(1.0)
        self.mean_rnd = tf.Variable(0.0)
        self.bettav = tf.Variable(1.0)

        self.mean_crt = [tf.Variable(1.0), tf.Variable(1.0)]
        self.mean_act = tf.Variable(1.0)

        self.cur_reward = 0.0
        self.max_reward = 1.0

        self.cur_reward100 = 0.0
        self.cur_reward10 = 0.0
        self.num_games = 0
        #trainable_val = self.modelq[0].trainable_variables + self.modelq[1].trainable_variables
        #self.grad_accum = [tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_val]

        self.this_pol= []

        self.nwin = "Main"
        #cv2.namedWindow(self.nwin)
        #cv2.setMouseCallback(self.nwin,self.capture_event)
        self.show_on = True
        self.xindex = []
        self.rewardy = []
        self.learn_index = 0
        self.max_index = 0
        self.cur_index = 0



    def capture_event(self,event,x,y,flags,params):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            self.show_on = not self.show_on
            print("St")


    @tf.function
    def get_net_res(self,l_state):
        inp = l_state
        out = self.modelp(inp , training = False)
        return out

    @tf.function
    def get_neta_res(self,l_state):
        inp = l_state
        out = self.modelpi(inp , training = False)
        index = tf.random.categorical(out,1)[0]
        out = tf.math.exp(out)
        return out, index

    @tf.function
    def get_netq_res(self,l_state):
        inp = l_state
        out = self.modelq[0](inp , training = False)
        index = tf.argmax(out)
        #index = tf.random.categorical(out,1)[0]
        #out = tf.math.exp(out)
        return out, index


    @tf.function
    def get_value_res(self,l_state):

        val = self.modelq[0](l_state, training = False)[1]
        return val

    def get_net_act(self,l_state):

        out = self.get_net_res(numpy.array([l_state]))[0].numpy()

        #index = numpy.random.choice(list(range(self.len_act)), p=out)
        index = get_ind_r(out)

        return index, out

    def calc_value(self, v, dones, rews):
        T = self.T
        vst = numpy.zeros((T,))
        vst[T-1] = v*(1-dones[T-1])
        for i in range(T-2,-1,-1):
            vst[i] = rews[i]+self.gamma*vst[i+1]*(1-dones[i])
        return vst

    @tf.function
    def train_q1(self, inp, inp_next, actn, rew, dones, pol):
        with tf.GradientTape(persistent=True) as tape1:

            qv, targ, q, qt  = [], [], [], []
            for i in range(self.nnets):
                qvl = self.modelq[i](inp, training = True)
                qvt = self.targetq[i](inp, training = True)
                qv.append(qvl)
                tqvl = self.targetq[i](inp_next, training = True)
                targ.append(tqvl)
                q.append(tf.gather_nd(batch_dims=1,params = qvl,indices  = actn))
                #qt.append(tf.gather_nd(batch_dims=1,params = qvt,indices  = actn))
                qt.append(qvt)


            targst = tf.stack(targ)
            minq = tf.reduce_min(targst,axis=0)
            maxq = tf.reduce_max(targst,axis=0)
            targst = tf.where(targst<0.0,targst+10000.0,targst)
            minq1 = tf.reduce_min(targst,axis=0)
            res = tf.where(maxq<0.0,maxq, minq1)
            minq = tf.where(minq<0,res,minq)

            qe = self.modelp(inp_next)
            minq2  = tf.reduce_sum(minq*qe,axis=-1)


            #yp = self.modelp(inp)
            log = tf.math.log(qe+1e-12)
            entr = tf.reduce_sum(- log*qe,axis=-1)
            #entr = tf.gather_nd(batch_dims=1,params = entr,indices  = actn)
            if(tf.random.uniform((1,),0,1)>0.999):
                tf.print(minq2)
                tf.print(tf.reduce_min(minq))


            qvt = rew + self.gamma*(minq2+entr*self.alphav)*(1-dones)



            dif = []
            for i in range(self.nnets):
                qtmin = tf.reduce_min(qt[i],axis=-1)
                coef  = tf.where(qtmin>0,1.0,0.0)[:,None]
                qtval = coef*qt[i]+(1-coef)*(qt[i]-qtmin[:,None])

                dif1a = tf.math.square(q[i]-qvt)#+tf.reduce_mean(tf.math.square(qv[i]-qtval)*1.0,axis=-1)*0.1
                #dif1a = dif1a + tf.reduce_mean(tf.math.square(tf.nn.relu(-2.0-qv[i])),axis=-1)*100.0
                #dif1b = tf.math.square(qt[i]+tf.clip_by_value(q[i]-qt[i],-self.border,self.border)-qvt)
                #dif1b=tf.reduce_mean(tf.maximum(dif1a,dif1b))
                dif.append(dif1a)


            lossq = tf.reduce_mean(tf.convert_to_tensor(dif,tf.float32))
            trainable_varsa = []
            for i in range(self.nnets):
                trainable_varsa.extend(self.modelq[i].trainable_variables)



        gradsa = tape1.gradient(lossq, trainable_varsa)
        self.optimizer1.apply_gradients(zip(gradsa, trainable_varsa))

        return lossq


    @tf.function
    def train_actor1(self, inp, inp_next, actn):
        with tf.GradientTape(persistent=True) as tape2:
            qv, qvt, alfa = [], [], []
            for i in range(self.nnets):
                qvl  = self.modelq[i](inp, training = True)
                qv.append(qvl)

            y_pii = self.modelp(inp, training = True)
            logpi = tf.math.log(y_pii+1e-12)
            entr = - tf.reduce_mean(tf.reduce_sum(y_pii*logpi, axis=-1))
            qvall = tf.stack(qv)
            minq = tf.reduce_min(qvall,axis=0)
            maxq = tf.reduce_max(qvall,axis=0)
            qvall = tf.where(qvall<0.0,qvall+10000.0,qvall)
            minq1 = tf.reduce_min(qvall,axis=0)
            res = tf.where(maxq<0.0,maxq, minq1)
            minq = tf.where(minq<0,res,minq)

            minq1 = minq

            maxx = tf.reduce_max(minq1,axis=-1)[:,None]
            qe = tf.exp((minq1-maxx)/self.alphav)

            qsum1 = tf.reduce_sum(qe,axis=-1)
            qe = (qe+1e-5) / (qsum1[:,None]+1e-5*self.len_act)
            log = tf.math.log(qe)
            #rel = tf.clip_by_value(y_pii/(qe+1e-10),5e-1,2.0)
            #dif1 = tf.reduce_sum(- qe*tf.math.log(rel),axis=-1)
            #dif1 = tf.reduce_sum(y_pii*(logpi-qe),axis=-1) #+ tf.reduce_sum(qe*(log-logpi),axis=-1)

            lossp =tf.reduce_mean(tf.reduce_sum(y_pii*(logpi-log),axis=-1)) #tf.reduce_mean(1 - tf.reduce_sum(tf.sqrt(qe*y_pii),axis=-1))

            trainable_vars2 = self.modelp.trainable_variables

        grads2 = tape2.gradient(lossp, trainable_vars2)
        self.optimizer2.apply_gradients(zip(grads2, trainable_vars2))



        with tf.GradientTape(persistent=True) as tape3:
            N = tf.cast(self.len_act,tf.float32)
            minq1 = minq/self.alphav
            qe = tf.math.exp(minq1 - tf.reduce_max(minq1,axis=-1)[:,None])
            qsum1 = tf.reduce_sum(qe,axis=-1)
            qe = qe / qsum1[:,None]
            qe = (qe+2e-10) / (1.0+2e-10*self.len_act)

            lN = tf.math.log(N)
            ent = lN - tf.reduce_mean(-qe*tf.math.log(qe),axis=-1)
            lossh = tf.reduce_mean(ent)
            lossh = tf.reduce_mean(tf.nn.relu(lossh-0.8*lN))

        #with tf.GradientTape(persistent=True) as tape3:

        #    N = tf.cast(self.len_act,tf.float32)
        #    minq1 = minq/(self.alphav*self.alpha_val)
        #    qe = tf.math.exp(minq1 - tf.reduce_max(minq1,axis=-1)[:,None])
        #    qsum1 = tf.reduce_sum(qe,axis=-1)
        #    qe = (qe+1e-9) / (qsum1[:,None] + 1e-9*N)
        #    lossh = - tf.reduce_mean(tf.reduce_sum(minq*tf.math.log(qe),axis=-1))
        #    lossh = lossh + tf.reduce_mean(tf.reduce_sum(qe*tf.math.log(qe),axis=-1))*0.01
        #    lossh = lossh + tf.nn.relu(0.2-self.alpha_val)*10.0+tf.nn.relu(self.alpha_val-10.0)*10.0
        #gradsb = tape3.gradient(lossh, [self.alpha_val])
        #self.optimizer3.apply_gradients(zip(gradsb, [self.alpha_val]))

        #with tf.GradientTape(persistent=True) as tape3:
        #    pp = self.modelpi(inp)
        #    lossh = - tf.reduce_mean(tf.reduce_sum(minq*tf.math.log(pp),axis=-1))
        #    lossh = lossh + tf.reduce_mean(tf.reduce_sum(pp*tf.math.log(pp),axis=-1))*0.01
        #    trainable_vars3 = self.modelpi.trainable_variables

        #with tf.GradientTape(persistent=True) as tape3:
        #    minq = tf.reduce_max(qvall,axis=0)
        #    minq1 = minq/(self.alphav)
        #    maxx = tf.reduce_max(minq1,axis=-1)[:,None]
        #    qe = tf.exp((minq1-maxx)/self.alphav)
        #    qsum1 = tf.reduce_sum(qe,axis=-1)
        #    qe = (qe+1e-12) / (qsum1[:,None]+1e-12*self.len_act)
        #    lossh = tf.nn.relu(tf.reduce_mean(tf.reduce_max(qe,axis=-1))-0.8)



        gradsb = tape3.gradient(lossh, [self.alphav])
        self.optimizer3.apply_gradients(zip(gradsb,[self.alphav]))
        return  lossp, tf.reduce_mean(entr) , lossp

    @tf.function
    def train_autoencoder(self, inp, inpnext, act):
         with tf.GradientTape(persistent=False) as tape:
             out = self.modelaenc([inp,act])
             difenc = tf.reduce_mean(tf.square(tf.cast(inpnext, tf.float32)/255.0-out),axis = [-2,-1])
             loss = tf.reduce_mean(difenc)

         trainable_vars = self.modelaenc.trainable_variables
         grads = tape.gradient(loss, trainable_vars)
         self.optimizera.apply_gradients(zip(grads, trainable_vars))
         return loss


    @tf.function
    def target_train(self,i):
        target_weights = self.targetq[i].trainable_variables
        weights = self.modelq[i].trainable_variables

        for (a, b) in zip(target_weights, weights):
            a.assign(b)


        return


    def learn_all(self):
        if self.flag_buf:
            max_count = self.n_buffer
        else:
            max_count = self.buf_index

        indices = numpy.random.choice(max_count, self.T)

        inp_next = tf.cast(self.states[indices] ,tf.uint8)
        inp = tf.cast(self.previous_states[indices],tf.uint8)

        acts = tf.cast(self.acts[indices] ,tf.int32)
        rews = tf.cast(self.rews[indices] ,tf.float32)
        dones = tf.cast(self.dones[indices] ,tf.float32)
        pol = tf.cast(self.policies[indices] ,tf.float32)

        lossq = self.train_q1(inp,inp_next,acts, rews, dones, pol)
        lossp, entr, dif = self.train_actor1(inp,inp_next,acts)

        #self.train_qauto(inp, inp_next, acts,dones)
        #self.train_pauto(inp)
        #if  (self.learn_index//50)%2==0 and self.learn_index%50<self.nnets:
        #    num_n = self.learn_index%self.nnets
        #    self.target_train(num_n)


        if(self.learn_index%20==0):
            self.modelpi.set_weights(self.modelp.get_weights())
            for vv in range(self.nnets):

                self.target_train(vv)
            #num_n = self.learn_index%self.nnets
            #print(num_n)
            #print(self.learn_index)


        self.learn_index += 1

        lossa = self.train_autoencoder(inp,inp_next,acts)

        return lossq, lossp, entr, lossa

    def add(self, reward,done,prev_state,state,act, pol):
        i = self.buf_index
        self.rews[i] = reward
        self.dones[i] = done
        self.states[i] = state
        self.previous_states[i] = prev_state
        self.acts[i] = act
        self.policies[i] = pol
        self.buf_index = self.buf_index+1
        if self.buf_index>=self.n_buffer:
            self.buf_index = 0
            self.flag_buf = True

        return


    def step(self):

        prev_st  = self.env.state()
        act, pol = self.get_net_act(prev_st)

        state, reward, done  = self.env.get_state(act)

        self.add(reward,done,prev_st,state,act, pol)

        self.cur_reward = self.cur_reward+reward

        if done:
            self.cur_index = self.env.index
            if self.env.index > self.max_index:
                self.max_index = self.env.index
            self.env.env_reset()
            self.cur_reward10 = self.cur_reward10 + self.cur_reward
            self.cur_reward = 0
            self.num_games = self.num_games + 1
            if self.num_games>100:
                self.cur_reward100 = self.cur_reward10/100
                val = self.cur_reward100
                self.xindex.append(self.index)
                self.rewardy.append(self.cur_reward100)

                self.cur_reward10 = 0
                self.num_games = 0

                if self.cur_reward100 > self.max_reward:
                    self.max_reward = self.cur_reward100
                    self.modelp_max.set_weights(self.modelp.get_weights())
                    #self.alphav.assign(0.01/self.max_reward)
                   

                inum = self.max_rewards.argmin()
                self.max_rewards[inum] = self.cur_reward100
                mean = self.max_rewards.max()

        if self.flag:
            self.show()

        if self.index>self.T*4 and self.index%64==0:

            lossq, lossp, entr, dif = self.learn_all()


            if(self.index%4000==0 and self.buf_index>self.T):

                inf_str = f"index {self.index}  maxind {self.max_index} cur_ind {self.cur_index} {self.alphav.numpy():.{2}e} maxrew {self.max_rewards} lossq {lossq:.{2}e}  lossp {lossp:.{2}e} entr {entr:.{2}e}  dif {dif:.{2}e} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100:.{3}f} "

                timed = (datetime.now() - self.start_time).total_seconds()

                #text_widget.value = inf_str +  f" time: {int(timed//3600)}:{int((timed%3600)//60)}"

                print(inf_str+f" time: {int(timed//3600)}:{int((timed%3600)//60)}")
                #print(self.this_pol)
                self.cur_reward = 0

                #self.show()
                if(self.index%32000==0):
                    plt.plot(self.xindex,self.rewardy)
                    plt.savefig("./out/figure_rew9.png")
                    plt.close()
                    df = pandas.DataFrame({'x': self.xindex, 'y': self.rewardy})
                    df.to_excel('./out/file_name10.xlsx', index=False)
                    #plotv(self.xindex,self.rewardy,self.env.get_image())



        self.index = self.index+1
        return

    def show(self):
        #frame = self.env.get_image()
        #plt.imshow(frame)

        return

env = environment()
sac1 = sac(env)
for i in range(100000000):
    sac1.step()
