
# MIT License
# Copyright (c) 2023 saysaysx
import random
import numpy
import time
import cv2
from random import choices
import tensorflow as tf
#import gymnasium as gym
import matplotlib.pyplot as plt
import pandas


import  gym


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

#tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization, Dropout, Conv2D, Reshape, Flatten
from tensorflow.keras.layers import MaxPooling2D, Permute, Conv1D, LSTM, GRU, LeakyReLU, Cropping1D, Multiply, Softmax, GaussianNoise
from tensorflow.keras.layers import RepeatVector, Subtract, Embedding, TimeDistributed, UpSampling2D
import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

def get_ind_r(x):
    CONST_MINV = 0.0011/(1-0.0011*len(x))
    x = x+CONST_MINV
    f = numpy.cumsum(x)/(1+CONST_MINV*len(x))
    v = numpy.random.random()
    index = numpy.digitize(v,f)
    if index>=len(x):
        index = len(x)-1
    return index



numpy.set_printoptions(precision=4)



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
import os
class environment():
    def __init__(self):
        #self.env = gym.make("Breakout-v4", render_mode="rgb_array")
        #self.env = gym.make("MsPacman-v4", render_mode="rgb_array")
        self.env = gym.make("Assault-v4", render_mode="rgb_array")
        #self.env = gym.make("Pong-ram-v0", render_mode="rgb_array")
        #self.env = gym.make("CrazyClimber-ram-v0", render_mode="rgb_array")
        #self.env = gym.make("CrazyClimber-ramDeterministic-v0", render_mode="rgb_array")
        #self.env = gym.make("Breakout-ramDeterministic-v0", render_mode="rgb_array")
        self.env = gym.wrappers.GrayScaleObservation(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env,shape = (96,96))
        self.env = gym.wrappers.FrameStack(self.env,num_stack=4)



        print(self.env.action_space.n)
        self.n_action = self.env.action_space.n
        print("Максимальная награда за действие:", self.env.reward_range)

        self.step = 0

        self.reward = 0.0
        self.index = 0
        print(self.env.observation_space)
        shx = (self.env.observation_space.high).shape
        print(shx)
        self.state_max = self.env.observation_space.high
        self.state_min = self.env.observation_space.low
        print(self.state_max.shape)

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

        self.field = next_observation

        self.reward = self.reward+reward
        self.index = self.index + 1

        return self.state(), reward, done

    def env_reset(self):
        self.index = 0
        self.reward = 0
        im = numpy.array(self.env.reset()[0])

        next_observation = im

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


from datetime import datetime


from io import BytesIO
buf = BytesIO()
plt.savefig(buf, format='png')



def plotv(x,y,im):

    #plt.plot(x,y)
    fig,ax = plt.subplots(1,2,figsize=(10,2))
    ax[0].imshow(im)
    ax[1].plot(x,y)


    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plt.close()


print("Declare function plot")



class sac:
    def __init__(self,env):
        self.start_time = datetime.now()

        self.env = env
        self.index = 0

        self.max_size = self.index
        self.flag = False

        self.shape_state = env.get_shape_state()

        print(self.shape_state)
        self.len_act = env.get_len_acts()
        self.gamma = 0.99
        self.alpha = tf.Variable(1.0)


        self.max_t = 10

        self.T = 32
        self.n_buffer = 40000
        self.buf_index  = 0
        self.flag_buf = False
        self.indT = tf.range(self.T)

        print("Make bufs")

        self.rews = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.rewards = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.maxrews = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.acts = numpy.zeros((self.n_buffer, 1),dtype=numpy.float32)
        self.policies = numpy.zeros((self.n_buffer, self.len_act),dtype=numpy.float32)
        self.values = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.previous_states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.dones = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.errors = numpy.ones((self.n_buffer,),dtype=numpy.float32)
        self.aerrors = numpy.ones((self.n_buffer,),dtype=numpy.float32)
        self.max_error = 1.0
        self.vars = [i for i in range(self.len_act)]

        print("Start make networks")
        from keras.layers import Lambda
        from tensorflow.keras.utils import register_keras_serializable
        @register_keras_serializable()
        def rescale(x):
            y =  tf.cast(x, tf.float32)/256
            y = tf.transpose(y, perm = [0,2,3,1,4])

            y = Reshape((self.shape_state[1],self.shape_state[2],self.shape_state[0]*self.shape_state[3])) (y)
            return y

        resc = Lambda(rescale)


        print("Shape")
        print(self.shape_state)
        self.nhist = 4
        self.acts_1 = tf.cast(numpy.zeros((self.T,self.nhist))[:,:]+self.len_act+0.001,tf.int32)


        inp1 = Input(shape = self.shape_state, name = 'inp1', dtype='uint8')
        lay = resc(inp1)
        lay = Conv2D(filters = 32, kernel_size = (8,8), strides = (4,4), activation = 'relu', padding = 'same') (lay)
        lay = Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), activation = 'relu', padding = 'same') (lay)
        lay = Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), activation = 'relu', padding = 'same') (lay)
        lay = Flatten() (lay)
        layo = Dense(512, activation = 'relu', name = 'layo') (lay)



        model_up = keras.Model(inputs=inp1, outputs=[layo])
        self.model_q_up = keras.models.clone_model(model_up)

        layv1 = Dense(self.len_act, activation = 'linear') (self.model_q_up.get_layer("layo").output)
        layv2 = Dense(self.len_act, activation = 'linear') (self.model_q_up.get_layer("layo").output)
        layv3 = Dense(self.len_act, activation = 'linear') (self.model_q_up.get_layer("layo").output)

        self.nnets = 2
        self.npnets = 2

        self.modelq = keras.Model(inputs=self.model_q_up.get_layer("inp1").output,
                                  outputs=[layv1, layv2, layv3])
        self.qnets = [self.modelq]
        for i in range(self.npnets-1):
            self.qnets.append(keras.models.clone_model(self.modelq))


        print("try to save model")
        tf.keras.utils.plot_model(self.modelq, to_file='./out/netq.png', show_shapes=True)


        @register_keras_serializable()
        def scale(x):
            qe = tf.nn.softmax(x)
            return qe
        sc = Lambda(scale)


        self.model_p_up = keras.models.clone_model(model_up)
        layp1 = Dense(self.len_act, activation="linear") (self.model_p_up.layers[-1].output)
        layp = sc(layp1)
        self.modelp = keras.Model(inputs=self.model_p_up.layers[0].output,
                                  outputs=layp)
        self.pnets = [self.modelp]

        for i in range(self.npnets-1):
            self.pnets.append(keras.models.clone_model(self.modelp))
        self.rews_pnets = [0.0]*self.npnets
        self.ipnets = 0

        self.targetq = keras.models.clone_model(self.modelq)
        self.tnets = [self.targetq]
        for i in range(self.npnets-1):
            self.tnets.append(keras.models.clone_model(self.targetq))

        self.nrewards  = 3
        self.max_rewards = numpy.array([0.0]*self.nrewards)


        tf.keras.utils.plot_model(self.targetq, to_file='./out/nettq1.png', show_shapes=True)
        tf.keras.utils.plot_model(self.modelp, to_file='./out/netp.png', show_shapes=True)

        self.optimizer1 = [tf.keras.optimizers.Adam(learning_rate=0.0002,clipnorm = 1.0) for i in range(self.npnets)]
        self.optimizer2 = [tf.keras.optimizers.Adam(learning_rate=0.0002,clipnorm = 1.0) for i in range(self.npnets)]
        self.train_actor = [self.train_actorf(self.optimizer2[i],self.pnets[i]) for i in range(self.npnets)]
        self.train_critic = [self.train_q(self.optimizer1[i], self.qnets[i]) for i in range(self.npnets)]

        self.alphav = tf.Variable(0.0005)
        self.coef_rew = tf.Variable(1.0)
        self.amaxrew = tf.Variable(-100000.0)
        self.aminrew = tf.Variable(100000.0)
        self.qvt_max = tf.Variable(-100000.0)
        self.alpha_val = tf.Variable(0.0)
        self.alphat = tf.constant([0.009,0.009,0.009,0.009,0.009])
        self.aut_loss = tf.Variable(0.000)

        self.border = tf.Variable(0.02)
        self.std_rnd = tf.Variable(1.0)
        self.mean_rnd = tf.Variable(0.0)
        self.bettav = tf.Variable(1.0)
        self.tettav = tf.Variable(1.0)
        self.ro = tf.Variable(200.0)

        self.mean_crt = [tf.Variable(1.0), tf.Variable(1.0)]
        self.mean_act = tf.Variable(1.0)

        self.cur_reward = 0.0
        self.max_reward = 1.0

        self.cur_reward100 = 0.0
        self.cur_reward10 = 0.0
        self.num_games = 0

        self.this_pol= []

        self.nwin = "Main"
        self.show_on = True
        self.xindex = []
        self.rewardy = []
        self.alpha_vals = []
        self.minq_vals = []
        self.cur_minq = tf.Variable(0.0)


        self.learn_index = 0
        self.max_index = 0
        self.cur_index = 0
        self.mean_index = 400

        self.maxbuf = []
        self.indbuf = []
        self.last_ind = 0
        self.count_steps = 0
        self.prev_index = 0
        self.new_index = 0
        self.num_learn = 16
        self.learn_target = 4

        self.nwin = "Main"
        cv2.namedWindow(self.nwin)
        cv2.setMouseCallback(self.nwin,self.capture_event)
        self.show_on = True

        self.CONST_MINV = 0.00101/(1-0.00101*self.len_act)

        #print(self.gammaval)






    def capture_event(self,event,x,y,flags,params):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            self.show_on = not self.show_on
            print("St")


    @tf.function
    def get_net_res(self,inet,l_state):
        inp = l_state
        out = self.pnets[inet](inp, training = False)
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
        out = self.get_net_res(self.ipnets,numpy.array([l_state]))[0].numpy()
        index = get_ind_r(out)
        return index, out

    def calc_value(self, v, dones, rews):
        T = self.T
        vst = numpy.zeros((T,))
        vst[T-1] = v*(1-dones[T-1])
        for i in range(T-2,-1,-1):
            vst[i] = rews[i]+self.gamma*vst[i+1]*(1-dones[i])
        return vst

    def train_q(self, optimizer, model):
        @tf.function
        def train_q1(rew, dones, acts, inp, inp_next):
            with tf.GradientTape(persistent=True) as tape1:
                tqvl = tf.convert_to_tensor(self.tnets[0]([inp_next], training = True))
                for i in range(1,self.npnets):
                    tqvl = tqvl + tf.convert_to_tensor(self.tnets[i]([inp_next], training = True))
                tqvl = tqvl/self.npnets


                minq = tf.minimum(tqvl[0], tqvl[1])
                minq = tf.clip_by_value(minq, 1e-8, 100000.0)
                qe = minq
                qsum1 = tf.reduce_sum(qe,axis=-1)
                qe = qe / (qsum1[:,None])
                minq2  = minq*qe
                minq2 = tf.reduce_sum(minq2, axis= -1)
                N = tf.cast(self.len_act,tf.float32)
                log = - tf.math.log(qe)
                entr = tf.reduce_sum(log*qe,axis=-1)/tf.math.log(N)

                qvt = minq2+self.alphav*entr
                dif = []
                qvt = rew + self.gamma*(qvt)*(1-dones)
                qvl = model(inp, training = True)
                q1 = tf.gather_nd(batch_dims=1,params = qvl[0],indices  = acts)
                q2 = tf.gather_nd(batch_dims=1,params = qvl[1],indices  = acts)
                q3 = tf.gather_nd(batch_dims=1,params = qvl[2],indices  = acts)
                dif.append(tf.square( q1 - qvt))
                dif.append(tf.square( q2 - qvt))
                dif.append(tf.square( q3 - rew))

                if tf.random.uniform((1,),0,1)>0.9995:
                    tf.print("minq-------------")
                    tf.print(minq2)
                    tf.print(tf.reduce_min(minq))
                    tf.print(tf.reduce_max(minq))
                    tf.print("rews------------")
                    tf.print(self.amaxrew)
                    tf.print(self.aminrew)

                errs = tf.reduce_mean(tf.stack(dif),axis=0)
                lossq = tf.reduce_mean(errs)
                trainable_varsa = model.trainable_variables



            gradsa = tape1.gradient(lossq, trainable_varsa)
            optimizer.apply_gradients(zip(gradsa, trainable_varsa))

            return lossq, errs
        return train_q1

    def train_actorf(self, optimizer, pmodel):
        @tf.function
        def train_actor1(inp):
            with tf.GradientTape(persistent=True) as tape2:
                qv, qvt, alfa = [], [], []

                qvl  = tf.convert_to_tensor(self.qnets[0](inp, training = True))
                for i in range(1,self.npnets):
                    qvl = qvl + tf.convert_to_tensor(self.qnets[i](inp, training = True))
                qvl = qvl / self.npnets

                for i in range(self.nnets):
                    qv.append(qvl[i])

                y_pii = pmodel(inp, training = True)
                logpi = tf.math.log(y_pii+1e-12)
                entr = - tf.reduce_sum(y_pii*logpi, axis=-1)

                minq = tf.reduce_min(tf.stack(qv),axis=0)

                minq1 = minq

                maxx = tf.reduce_max(minq1,axis=-1)
                minx = tf.reduce_min(minq1,axis=-1)

                mm = maxx-minx
                alb = self.ro
                alpha = (mm+1.0)/(alb)
                qe = tf.exp((minq1-maxx[:,None])/alpha[:,None])


                qsum1 = tf.reduce_sum(qe,axis=-1)
                qe = qe / (qsum1[:,None])
                qe = (qe+self.CONST_MINV) / (1.0+self.CONST_MINV*self.len_act)

                log = tf.math.log(qe)

                err = tf.reduce_sum(qe*(log-logpi),axis=-1)
                lossp = tf.reduce_mean(err)

                trainable_vars2 = pmodel.trainable_variables

            grads2 = tape2.gradient(lossp, trainable_vars2)
            optimizer.apply_gradients(zip(grads2, trainable_vars2))

            return  lossp, tf.reduce_mean(entr), tf.reduce_max(minq)
        return train_actor1

    @tf.function
    def target_train(self):
        for i in range(self.npnets):
            target_weights = self.tnets[i].trainable_variables
            weights = self.qnets[i].trainable_variables
            tau  = 0.99
            for (a, b) in zip(target_weights, weights):
                a.assign(b*(1-tau)+a*tau)
        return


    def get_indicies(self,nhist,indicies):
        aind = numpy.ones((len(indicies),nhist))
        aind[:,0] = indicies
        aind = numpy.cumsum(aind,axis=1).astype(int)
        return aind


    def learn_all(self):
        if self.flag_buf:
            max_count = self.n_buffer
        else:
            max_count = self.buf_index


        indices = numpy.random.choice(max_count-self.nhist, self.T)
        acts = tf.cast(self.acts[indices], tf.int32)
        inp = tf.cast(self.previous_states[indices] ,tf.float32)
        inp_next = tf.cast(self.states[indices] ,tf.float32)
        rw = self.rews[indices]
        rw = numpy.clip(rw,-1,1)
        rews = tf.cast(rw ,tf.float32)
        dones = tf.cast(self.dones[indices] ,tf.float32)



        lossq, errs = self.train_critic[self.learn_index%self.npnets](rews, dones, acts, inp, inp_next)


        indices = numpy.random.choice(max_count, self.T)
        inp = tf.cast(self.states[indices],tf.uint8)
        lossp, entr, dif = self.train_actor[self.learn_index%self.npnets](inp)



        if(self.learn_index%10==0):

            self.alphav.assign(self.alphav.numpy()*1.0000001)
            if(self.alphav>7e-3):
                self.alphav.assign(7e-3)


            self.optimizer1[0].learning_rate = self.optimizer1[0].learning_rate*0.99999
            for iopt in range(self.npnets):
                self.optimizer1[iopt].learning_rate = self.optimizer1[0].learning_rate
                self.optimizer2[iopt].learning_rate = self.optimizer1[0].learning_rate*0.99999


            if(self.optimizer1[0].learning_rate<2e-6):
                self.optimizer1[0].learning_rate = 2e-6


        if(self.learn_index%self.learn_target==0):
            self.target_train()

        self.learn_index += 1

        return  lossq, lossp, entr, dif

    def add(self, reward,done,prev_state,state,act, pol, rewards):
        i = self.buf_index
        self.rews[i] = reward
        self.rewards[i] = rewards
        self.maxrews[i] = self.amaxrew if self.amaxrew>rewards else rewards
        self.dones[i] = done
        self.states[i] = state
        self.previous_states[i] = prev_state
        self.acts[i] = act
        self.policies[i] = pol
        self.errors[i] = 100.0
        self.aerrors[i] = 100.0


        self.buf_index = self.buf_index+1
        if self.buf_index>=self.n_buffer:
            self.buf_index = 0
            self.flag_buf = True
            self.amaxrew.assign(self.rewards.max())
            self.aminrew.assign(self.rewards.min())



        if self.amaxrew<rewards:
            self.amaxrew.assign(rewards)
        if self.aminrew>rewards:
            self.aminrew.assign(rewards)
        self.count_steps += 1
        if done:
            self.prev_index = self.new_index
            self.new_index = i
            if self.new_index>self.prev_index:
                self.maxrews[self.prev_index+1:self.new_index] = rewards
            else:
                self.maxrews[self.prev_index+1:] = rewards
                self.maxrews[0:self.new_index] = rewards

            self.count_steps = 0

        return


    def step(self):

        prev_st  = self.env.state()
        act, pol = self.get_net_act(prev_st)

        state, reward, done  = self.env.get_state(act)

        self.add(reward,done,prev_st,state,act, pol, self.env.reward)

        self.cur_reward = self.cur_reward+reward

        if done:
            self.rews_pnets[self.ipnets] = self.rews_pnets[self.ipnets]*0.9+self.cur_reward*0.1
            self.ipnets = self.ipnets + 1
            if(self.ipnets>=self.npnets):
                self.ipnets = 0


            self.cur_index = self.env.index
            self.mean_index = self.mean_index*0.95 + self.cur_index * 0.05

            if self.env.index > self.max_index:
                self.max_index = self.env.index
            self.env.env_reset()
            self.cur_reward10 = self.cur_reward10 + self.cur_reward
            self.cur_reward = 0
            self.num_games = self.num_games + 1
            if self.num_games>100:
                self.cur_reward100 = self.cur_reward10/100
                self.xindex.append(self.index)
                self.rewardy.append(self.cur_reward100)
                self.alpha_vals.append(self.alphav.numpy())
                self.minq_vals.append(self.cur_minq)

                self.cur_reward10 = 0
                self.num_games = 0

                if self.cur_reward100 > self.max_reward:
                    self.max_reward = self.cur_reward100

                    self.bettav.assign(0.95)
                else:
                    self.bettav.assign(0.9)

                inum = self.max_rewards.argmin()
                self.max_rewards[inum] = self.cur_reward100

                self.num_learn = int((self.mean_index/400))+1
                self.learn_target = int(self.mean_index/100)+1



        if self.flag:
            self.show()

        if self.index>self.T*4 and self.index%self.num_learn==0:


            lossq, lossp, entr, dif = self.learn_all()
            self.cur_minq = dif.numpy()

            if(self.index%(1000*self.num_learn)==0 and self.buf_index>self.T):

                inf_str = f"index {self.index}  maxind {self.max_index} cur_ind {self.cur_index} {self.alphav.numpy():.{2}e} maxrew {self.max_rewards} lossq {lossq:.{2}e}  lossp {lossp:.{2}e} entr {entr:.{2}e}  dif {dif:.{2}e} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100:.{3}f} "

                timed = (datetime.now() - self.start_time).total_seconds()

                print(inf_str+f"mean_index {self.mean_index} pnetrew {numpy.array(self.rews_pnets)}  ro {self.ro.numpy():.{4}e} lr {self.optimizer1[0].learning_rate.numpy():.{4}e} time: {int(timed//3600)}:{int((timed%3600)//60)}")
                self.show()

                if(self.index%(1000*self.num_learn)==0):
                    plt.plot(self.xindex,self.rewardy)
                    plt.savefig("./out/figure_rew9.png")
                    plt.close()
                    df = pandas.DataFrame({'x': self.xindex, 'y': self.rewardy, 'alpha':self.alpha_vals, 'minq':self.minq_vals})
                    df.to_excel('./out/file_name10.xlsx', index=False)

        self.index = self.index+1
        return

    def show(self):
        cv2.imshow(self.nwin,self.env.get_image())
        if self.flag:
            time.sleep(0.05)
        if cv2.waitKey(1)==27:
            print("27")
            self.flag = not self.flag
        return


env = environment()
sac1 = sac(env)
for i in range(100000000):
    sac1.step()
