
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

#tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization, Dropout, Conv2D, Reshape, Flatten
from tensorflow.keras.layers import MaxPooling1D, Permute, Conv1D, LSTM, LeakyReLU, Cropping1D, Multiply, Softmax, GaussianNoise
from tensorflow.keras.layers import RepeatVector, Subtract, Embedding
import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


numpy.set_printoptions(precision=4)

print("Start gput opts")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)
sess.as_default()


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
        if self.index>10000:
            done = True


        return self.state(), reward, done

    def env_reset(self):
        self.index = 0
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

        print("Shape")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state,  dtype='uint8')

        lay = resc(inp1)

        lay = Dense(350, activation = 'relu') (lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = Dense(150, activation = 'relu') (lay)
        lay = Dense(50, activation = 'relu') (lay)
        layv1 = Dense(self.len_act, activation = 'linear') (lay)


        self.nnets = 2
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
            y = x+5e-3
            sumy = tf.reduce_sum(y,axis=-1)[:,None]
            y = y/sumy
            return y

        sc = Lambda(scale)

        inp1 = Input(shape = self.shape_state, dtype='uint8' )

        lay = resc(inp1)
        coefv = 1e-3*self.len_act

        lay = Dense(350, activation = 'relu') (lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = Dense(150, activation = 'relu') (lay)
        lay = Dense(50, activation = 'relu') (lay)
        layp1 = Dense(self.len_act, activation="softmax") (lay)
        layp1 =  sc(layp1)


        self.modelp = keras.Model(inputs=inp1, outputs=layp1)


        self.targetq = [keras.models.clone_model(self.modelq[i]) for i in range(self.nnets)]

        self.nrewards  = 3
        self.max_rewards = numpy.array([0.0]*self.nrewards)


        tf.keras.utils.plot_model(self.targetq[0], to_file='./out/nettq1.png', show_shapes=True)
        tf.keras.utils.plot_model(self.modelp, to_file='./out/netp.png', show_shapes=True)

        self.hash_len = 32
        inp1 = Input(shape = self.shape_state, dtype='uint8' )
        lay = resc(inp1)


        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.optimizer4 = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.alphav = tf.Variable(0.003)
        self.border = tf.Variable(0.1)
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
        trainable_val = self.modelq[0].trainable_variables + self.modelq[1].trainable_variables
        self.grad_accum = [tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_val]



        self.nwin = "Main"
        #cv2.namedWindow(self.nwin)
        #cv2.setMouseCallback(self.nwin,self.capture_event)
        self.show_on = True
        self.xindex = []
        self.rewardy = []



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
    def get_value_res(self,l_state):

        val = self.modelq[0](l_state, training = False)[1]
        return val

    def get_net_act(self,l_state):
        out = self.get_net_res(numpy.array([l_state]))
        out1 = out[0].numpy()
        index = numpy.random.choice(self.len_act, 1, p = out1) [0]

        return index, out1


    # try rew + g*(1-down)*q() + H(s), opposite to rew + g*(1-down)*(q()+H(s_next))
    @tf.function
    def train_q1(self, inp, inp_next, actn, rew, dones, pol):
        with tf.GradientTape(persistent=True) as tape1:
            qv, targ, tqv  = [], [], []

            for i in range(self.nnets):
                qvl = self.modelq[i](inp, training = True)
                qv.append(qvl)
                val = self.targetq[i](inp_next, training = True)
                targ.append(val)
                tqv.append(self.targetq[i](inp, training = True))

            #y_pi = self.modelp(inp_next, training = True)
            #log = tf.math.log(y_pi)
            #pol = tf.clip_by_value(pol,1e-10,1.0)
            logpol = tf.math.log(pol)
            q, qt = [], []

            for i in range(self.nnets):
                q.append(tf.gather_nd(batch_dims=1,params = qv[i],indices  = actn))
                qt.append( tf.gather_nd(batch_dims=1,params = tqv[i],indices  = actn))

            minq  = tf.minimum(targ[0] , targ[1])

            minq1 = minq/self.alphav
            qe = tf.math.exp(minq1 - tf.reduce_max(minq1,axis=-1)[:,None])
            qsum1 = tf.reduce_sum(qe,axis=-1)
            qe = qe / qsum1[:,None]
            qe = tf.clip_by_value(qe,1e-3,1.0)
            log = tf.math.log(qe)

            acts = tf.random.categorical(logpol,1)
            acts1 = tf.random.categorical(log,1)

            nentr = self.alphav*logpol
            nentr = tf.gather_nd(batch_dims=1,params = nentr,indices  = acts)
            #kl = y_pii*(logpi-logpol)

            #minq1  = tf.reduce_sum(y_pi*minq,axis=-1)
            minq1  = tf.gather_nd(batch_dims=1,params = minq,indices  = acts1)


            qvt = tf.nn.softsign(rew) + self.gamma*minq1*(1-dones)  - nentr

            dif = []
            val = []
            dif = []
            for i in range(2):
                dif1a = tf.math.square(q[i]-qvt)
                #dif1b = tf.math.square(qt[i]+tf.clip_by_value(q[i]-qt[i],-self.border,self.border)-qvt)
                #dif.append(tf.reduce_mean(tf.maximum(dif1a,dif1b)))
                dif.append(tf.reduce_mean(dif1a))

            lossq = tf.reduce_mean(tf.convert_to_tensor(dif,tf.float32))
            trainable_varsa = self.modelq[0].trainable_variables + self.modelq[1].trainable_variables
            #trainable_varsb = self.rnd_model.trainable_variables

        gradsa = tape1.gradient(lossq, trainable_varsa)
        self.grad_accum = [acc.assign(acc+g) for acc, g in zip(self.grad_accum, gradsa)]
        return lossq

    @tf.function
    def apply_grads(self):
        trainable_varsa = self.modelq[0].trainable_variables + self.modelq[1].trainable_variables
        grads = [grad/4.0 for grad in self.grad_accum]
        self.optimizer1.apply_gradients(zip(grads, trainable_varsa))
        for acc in self.grad_accum:
            acc.assign(tf.zeros_like(acc))


    @tf.function
    def train_actor1(self, inp, pol):
        with tf.GradientTape(persistent=True) as tape2:

            qv, qvt = [], []
            for i in range(self.nnets):
                qv.append(self.modelq[i](inp, training = True))
                #qvt.append(self.targetq[i](inp, training = True))
            y_pii = self.modelp(inp, training = True)
            logpi = tf.math.log(y_pii)

            entr = - tf.reduce_mean(tf.reduce_sum(y_pii*logpi, axis=-1))
            #minq = tf.minimum(qv[0],qv[1])
            minq = (qv[0]+qv[1])*0.5
            #minq1 = minq/self.alphav
            #qe = tf.math.exp(minq1 - tf.reduce_max(minq1,axis=-1)[:,None])
            #qsum1 = tf.reduce_sum(qe,axis=-1)
            #qe = qe / qsum1[:,None] + 1e-9
            #qsum1 = tf.stop_gradient(tf.reduce_sum(qe,axis=-1))
            #qe = qe / qsum1[:,None]
            #qe = qe*0.7+pol*0.3

            #qelog = tf.math.log(qe)

            dif1 = tf.reduce_sum(y_pii*(logpi*self.alphav-minq) ,axis=-1)

            lossp = tf.reduce_mean(dif1)

            trainable_vars2 = self.modelp.trainable_variables

        grads2 = tape2.gradient(lossp, trainable_vars2)
        self.optimizer2.apply_gradients(zip(grads2, trainable_vars2))


        return  lossp, tf.reduce_mean(entr) , lossp


    @tf.function
    def target_train(self):
        tau = [0.003,0.003]
        for i in range(self.nnets):
            target_weights = self.targetq[i].trainable_variables
            weights = self.modelq[i].trainable_variables
            for (a, b) in zip(target_weights, weights):
                a.assign(a * (1-tau[i]) + b*tau[i])
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
        if self.step_s():
            self.apply_grads()

        lossp, entr, dif = self.train_actor1(inp, pol)
        #time.sleep(0.01)

        self.target_train()
        return lossq, lossp, entr, dif

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


                inum = self.max_rewards.argmin()
                self.max_rewards[inum] = self.cur_reward100
                mean = self.max_rewards.max()

        if self.flag:
            self.show()

        if self.index>self.T*4 and self.index%64==0:

            lossq, lossp, entr, dif = self.learn_all()


            if(self.index%4000==0 and self.buf_index>self.T):

                inf_str = f"index {self.index} {self.alphav.numpy():.{2}e} maxrew {self.max_rewards} lossq {lossq:.{2}e}  lossp {lossp:.{2}e} entr {entr:.{2}e}  dif {dif:.{2}e} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100:.{3}f} "

                timed = (datetime.now() - self.start_time).total_seconds()

                #text_widget.value = inf_str +  f" time: {int(timed//3600)}:{int((timed%3600)//60)}"
                #print(f"index {self.index} {self.alphav.numpy():.{2}e} maxrew {self.max_rewards} lossq {lossq:.{2}e}  lossp {lossp:.{2}e} entr {entr:.{2}e}  dif {dif:.{2}e} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100:.{3}f} ")
                print(inf_str+f" time: {int(timed//3600)}:{int((timed%3600)//60)}")
                self.cur_reward = 0

                #self.show()
                if(self.index%32000==0):
                    plt.plot(self.xindex,self.rewardy)
                    plt.savefig("./out/figure_rew8.png")
                    plt.close()
                    df = pandas.DataFrame({'x': self.xindex, 'y': self.rewardy})
                    df.to_excel('./out/file_name8.xlsx', index=False)
                    plotv(self.xindex,self.rewardy,self.env.get_image())


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
