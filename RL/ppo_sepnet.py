# MIT License
# Copyright (c) 2023 saysaysx
import random

#I can’t understand why this code is so slow compared to those given in the articles,
# training requires more than 100 million received frames, while memory consumption
# increases and intermediate saving and running the code again is required.
# Memory consumption grows due to changes in the size of the shape of the supplied arrays,
# but here an attempt was made to minimize such situations as much as possible,
# but memory growth still occurs, and, nevertheless, a significant consumption of game situations.

import numpy
import time

from random import choices
import tensorflow as tf
import gym
import cv2
import matplotlib.pyplot as plt

#import atari_py
#from ale_py import ALEInterface
#ale = ALEInterface()
#from ale_py.roms import Breakout
#ale.loadROM(Breakout)


import os
#os.environ['ROMS_PATH'] = '../../sii1/atari/ROMS'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization, Dropout, Conv2D, Reshape, Flatten
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


#tf.config.run_functions_eagerly(True)
sess = tf.compat.v1.Session(config=config)
sess.as_default()

#import stable_baselines3.common.atari_wrappers as wr


class environment():
    def __init__(self):
        self.env = gym.make("Breakout-v4")

        #self.env = wr.AtariWrapper(self.env,frame_skip=1, terminal_on_life_loss=False)
        #self.env = wr.NoopResetEnv(self.env)
        #self.env = wr.FireResetEnv(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env,shape=(84,84))
        self.env = gym.wrappers.GrayScaleObservation(self.env,keep_dim=True)
        self.env = gym.wrappers.FrameStack(self.env,num_stack=4)




        n_action = self.env.action_space.n
        self.step = 0
        self.acts = numpy.zeros(n_action)
        self.acts[0] = 1.0
        self.reward = 0
        self.index = 0

        self.state_max = self.env.observation_space.high.max()
        self.state_min = self.env.observation_space.low.min()


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
            self.reward  = 0


        self.ind_obs = 0

        next_observation, reward, done, _, info = self.env.step(act)

        self.field = numpy.transpose(numpy.array(next_observation).squeeze(),(1,2,0))



        self.reward = self.reward+reward
        self.index = self.index + 1



        return done, self.state(), reward

    def env_reset(self):
        self.index = 0
        self.env.reset()
        im = self.env.frames
        next_observation = numpy.array(im)
        self.field = numpy.transpose(next_observation.squeeze(),(1,2,0))



    def get_image(self):
        return self.env.render(mode='rgb_array')

    def state(self):
        state = self.field/self.state_max

        return state#(( state - self.state_min) / (self.state_max - self.state_min)-0.5)*2



    def get_len_state(self):
        return len(self.state())


class multi_environment:
    def __init__(self,n = 1):
        self.N = n
        self.envs = [environment() for i in range(n)]

    def get_shape_state(self):
        return self.envs[0].state().shape
    def get_len_acts(self):
        return len(self.envs[0].acts)

    def get_step(self, i, act):
        return self.envs[i].get_state(act)

    def get_cur_state(self, i):
        return self.envs[i].state()

    def reset(self, i):
        return self.envs[i].env_reset()


    def get_image(self,i):
        return self.envs[i].get_image()


class ppo:
    def __init__(self,envs):
        self.envs = envs
        self.index = 0

        self.max_size = self.index
        self.flag = False

        self.shape_state = envs.get_shape_state()
        self.len_act = envs.get_len_acts()
        self.gamma = 0.99


        self.max_t = 10
        self.start_time = time.time()

        self.T = 128
        self.N = envs.N


        self.rews = numpy.zeros((self.N, self.T),dtype=numpy.float32)
        self.acts = numpy.zeros((self.N, self.T),dtype=numpy.int32)
        self.policies = numpy.zeros((self.N,self.T, self.len_act),dtype=numpy.float32)
        self.values = numpy.zeros((self.N,self.T),dtype=numpy.float32)
        self.states = numpy.zeros((self.N, self.T, *self.shape_state),dtype=numpy.float32)
        self.previous_states = numpy.zeros((self.N, self.T, *self.shape_state),dtype=numpy.float32)
        self.dones = numpy.zeros((self.N,self.T),dtype=numpy.float32)
        self.cur_rewards = numpy.zeros((self.N),dtype=numpy.int32)
        self.NSTEP = 128
        self.NSTEPR = 256
        self.all_rewards = numpy.zeros((self.N, self.NSTEP),dtype=numpy.int32)
        self.all_dif_rewards = numpy.zeros((self.NSTEPR),dtype=numpy.int32)
        self.cur_dif_step = 0
        self.cur_step = numpy.zeros((self.N),dtype=numpy.int32)
        self.rewards_step = [[0] for i in range(self.N)]


        inp = Input(shape = self.shape_state)
        lay = Conv2D(16,(8,8), strides = (4,4),activation= 'relu',padding='same') (inp)
        lay = Conv2D(32,(4,4), strides = (2,2), activation= 'relu',padding='same') (lay)
        lay1v = Conv2D(64,(3,3), strides = (2,2), activation= 'relu',padding='same') (lay)
        lay1 = Flatten() (lay1v)
        self.modelA1 = keras.Model(inputs=inp, outputs=[lay1])

        lay = Conv2D(16,(8,8), strides = (4,4),activation= 'relu',padding='same') (inp)
        lay = Conv2D(32,(4,4), strides = (2,2), activation= 'relu',padding='same') (lay)
        lay2v = Conv2D(64,(3,3), strides = (2,2), activation= 'relu',padding='same') (lay)
        lay2 = Flatten() (lay2v)
        self.modelB1 = keras.Model(inputs=inp, outputs=[lay2])

        lay = Conv2D(16,(8,8), strides = (4,4),activation= 'relu',padding='same') (inp)
        lay = Conv2D(32,(4,4), strides = (2,2), activation= 'relu',padding='same') (lay)
        lay3v = Conv2D(64,(3,3), strides = (2,2), activation= 'relu',padding='same') (lay)




        def dconv(filt,size,x, strides=2):
            y = keras.layers.UpSampling2D(size=strides)(x)
            y = keras.layers.Conv2D(filters = filt,kernel_size=size,padding='same') (y)
            return y

        layc = dconv(64,3,lay3v,strides=2)
        layc = dconv(32,4,layc,strides=2)
        layc = dconv(4,8,layc,strides=4)
        layc = keras.layers.Conv2D(filters = 4,kernel_size=5,padding='same',activation='tanh') (layc)
        layc = keras.layers.Cropping2D(cropping=((6, 6), (6, 6))) (layc)
        self.modelC = keras.Model(inputs=inp, outputs=[layc])


        layc3 = Flatten() (lay3v)
        layc3 = Dense(128, activation = 'relu') (layc3)

        layc1 = Dense(128, activation = 'relu') (lay1)
        layc2 = Dense(128, activation = 'relu') (lay2)

        layc1 = keras.layers.Concatenate() ([layc1,layc3])
        layc2 = keras.layers.Concatenate() ([layc2,layc3])


        layv = Dense(1, activation = 'linear') (layc1)
        layp = Dense(self.len_act, activation="softmax") (layc2)
        self.modelA2 = keras.Model(inputs=inp, outputs=[layp])
        self.modelB2 = keras.Model(inputs=inp, outputs=[layv])

        self.model = keras.Model(inputs=inp, outputs=[layp, layv, layc])

        tf.keras.utils.plot_model(self.model, to_file='./out/actor_critic_s.png', show_shapes=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.rates = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
        self.entr_num = 0
        self.learn_num = 0


        self.learn_rate = 1.0
        self.c_entr = 1.0


        self.n_models = 5
        self.learn_nums = [i for i in range(self.n_models)]
        self.entr_nums = [i for i in range(self.n_models)]

        self.models = [keras.models.clone_model(self.model)  for i in range(self.n_models)]
        self.max_rewards = numpy.zeros((self.n_models))
        self.models_time = numpy.zeros((self.n_models))

        #self.model.load_weights("./out/name2.h5")


        self.nwin = "Main"
        cv2.namedWindow(self.nwin)
        cv2.setMouseCallback(self.nwin,self.capture_event)
        self.show_on = True

    def capture_event(self,event,x,y,flags,params):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            self.show_on = not self.show_on
            print("St")


    @tf.function
    def get_net_res(self,l_state):
        out = self.model(l_state, training = False)
        return out

    @tf.function
    def get_value_res(self,l_state):
        val = self.model(l_state, training = False)[1]
        return val

    def get_net_act(self,l_state):

        out = self.get_net_res(l_state)

        vars = [i for i in range(self.len_act)]
        index = numpy.array([choices(vars, prob)[0] for prob in out[0]])

        return out[0], index, out[1]



    def calc_value(self, vcurrent):
        T = self.T
        N = self.N
        vst = numpy.zeros((N, T))
        vst[:,T-1] = vcurrent[:,T-1]*(1-self.dones[:,T-1])
        for i in range(T-2,-1,-1):
            vst[:,i] = self.rews[:,i]+self.gamma*vst[:,i+1]*(1-self.dones[:,i])

        return vst

    def calc_value_delta(self, vcurrent):
        T = self.T
        N = self.N
        vst = numpy.zeros((N, T))
        vst[:,T-1] = vcurrent[:,T-1]*(1-self.dones[:,T-1])
        vst[:,0:T-1] = self.rews[:,0:T-1]+self.gamma*vcurrent[:,1:T]*(1-self.dones[:,0:T-1])
        return vst

    def calc_advantage_gae(self, vst, value):
        T = self.T
        sig = vst - value
        adv = numpy.zeros(sig.shape)

        lam = 0.95
        adv[:,T-1] = sig[:,T-1]
        for i in range(T-2, -1, -1):
            adv[:,i] = sig[:,i] + lam*self.gamma*adv[:,i+1]*(1-self.dones[:,i])

        return adv



    def calc_advantage(self, vst, value):
        adv = vst - value
        return adv





    @tf.function
    def train_actor(self, inp, adv, acts, pol, vst, value_old):
        with tf.GradientTape(persistent=True) as tape:
            y_pi, v, im = self.model(inp, training = True)
            #y_pin, _ = self.modelm(inp, training = False)

            actn  = tf.reshape(acts, (acts.shape[0],1))
            advm = tf.reduce_mean(adv)
            adv2 = tf.sqrt(tf.reduce_mean((adv-advm)**2))
            advs = tf.divide(adv-advm, adv2+1e-8)
            y_pi2 = tf.gather_nd(batch_dims=1,params = y_pi,indices  = actn)
            y_old = tf.gather_nd(batch_dims=1,params = pol,indices  = actn)

            rel = tf.math.exp(tf.math.log(y_pi2+1e-12)-tf.math.log(y_old+1e-12))

            relclip = tf.clip_by_value(rel,0.9,1.1)

            relmin = tf.minimum(rel*advs,relclip*advs)
            loss_value = -tf.reduce_sum(relmin)
            entr = tf.reduce_sum(y_pi*tf.math.log(y_pi+1e-8),axis=1)
            kb = tf.math.reduce_max(tf.reduce_sum((y_pi-pol)*tf.math.log((y_pi)/(pol)),axis=1))
            #kb = tf.abs(tf.reduce_mean(tf.math.log(pol/y_pi)))
            #ls = tf.reduce_mean(tf.reduce_sum(y_pi*(tf.math.log(y_pi+1e-12)-tf.math.log(y_pin+1e-12)), axis=1))
            val = value_old+ tf.clip_by_value(v - value_old,-22.5,22.5)

            loss_vst1 = (vst - v)**2
            loss_vst2 = (vst - val)**2
            loss_vst = tf.reduce_sum(tf.maximum(loss_vst1, loss_vst2))
            loss_entr = tf.reduce_sum(entr)

            loss_valueA = loss_value + 0.01*self.c_entr*loss_entr #+ 0.01*ls
            loss_valueB = loss_vst

            loss_valueC = tf.reduce_mean(tf.square(im-inp))

            trainable_varsA = self.modelA1.trainable_variables+self.modelA2.trainable_variables
            trainable_varsB = self.modelB1.trainable_variables+self.modelB2.trainable_variables
            trainable_varsC = self.modelC.trainable_variables

        gradsA = tape.gradient(loss_valueA, trainable_varsA)
        gradsB = tape.gradient(loss_valueB, trainable_varsB)
        gradsC = tape.gradient(loss_valueC, trainable_varsC)
        #grads, gnorm = tf.clip_by_global_norm(grads, 5.0)

        self.optimizer.apply_gradients(zip([grad for grad in gradsA], trainable_varsA))
        self.optimizer.apply_gradients(zip([grad for grad in gradsB], trainable_varsB))
        self.optimizer.apply_gradients(zip([grad for grad in gradsC], trainable_varsC))

        return loss_value, loss_entr, loss_vst, kb

    def max_num(self,rews):
        r = rews.copy()
        vals = []
        for i in range(len(rews)//2):
            mi = r.argmax()
            vals.append(mi)
            r[mi] = -0.1

        mj = int(numpy.random.random()*len(vals))
        return vals[mj]


    def learn_all(self):
        T = self.T
        N = self.N


        pstates = self.previous_states.reshape(N*T, *self.shape_state)


        vst = self.calc_value_delta(self.values)
        adv1 = self.calc_advantage_gae(vst, self.values)
        vst = adv1 + vst

        adv = adv1.reshape(N*T)
        acts = self.acts.reshape(N*T)
        pol = self.policies.reshape(N*T, self.len_act)
        vst = vst.reshape(N*T)
        val_old = self.values.reshape(N*T)

        EP = 16
        S = N*T//EP


        nc =  int(len(self.max_rewards)*numpy.random.random())

        for i in range(16):
            #int(numpy.random.random()*self.n_models)

            self.modelm = self.models[nc]

            j = i#int(numpy.random.random()*EP)
            index = list(range(j*S,S*(j+1)))
            st_c = tf.stop_gradient(tf.cast(pstates[index] ,tf.float32))
            adv_c = tf.stop_gradient(tf.cast(adv[index] ,tf.float32))
            acts_c = tf.stop_gradient(tf.cast(acts[index] ,tf.int32))
            pol_c = tf.stop_gradient(tf.cast(pol[index] ,tf.float32))
            vst_c = tf.stop_gradient(tf.cast(vst[index] ,tf.float32))
            val_old_c = tf.stop_gradient(tf.cast(val_old[index] ,tf.float32))

            #xw = self.model.get_weights()
            _,_,_, ret = self.train_actor(st_c, adv_c, acts_c, pol_c, vst_c, val_old_c)

            #xw = self.model.get_weights()
            #if ret>0.001:
            #    self.model.set_weights(xw)



        return

    def step(self):
        for j in range(self.T):
            for i in range(self.N):
                self.previous_states[i,j] = envs.get_cur_state(i)

            policy, act, value = self.get_net_act(self.previous_states[:,j])



            for i in range(self.N):
                done, state, reward = envs.get_step(i, act[i])


                self.rews[i,j] = reward
                self.dones[i,j] = done
                self.policies[i,j] = policy[i]
                self.values[i,j] = value[i]
                self.states[i,j] = state
                self.acts[i,j] = act[i]
                self.cur_rewards[i] = self.cur_rewards[i]+reward
                if done:
                    envs.reset(i)
                    self.all_rewards[i,self.cur_step[i]] = self.cur_rewards[i]
                    self.all_dif_rewards[self.cur_dif_step] = self.cur_rewards[i]

                    self.cur_rewards[i] = 0
                    self.cur_step[i]=self.cur_step[i]+1

                    self.cur_dif_step+=1
                    if(self.cur_dif_step>=self.NSTEPR):

                        mini = self.max_rewards.argmin()
                        maxi = self.max_rewards.argmax()
                        rewm = self.all_dif_rewards.mean()





                        self.entr_nums[mini] = self.entr_num
                        self.learn_nums[mini] = self.learn_num

                        num1 = self.max_rewards.argmax()
                        num2 = int(len(self.max_rewards)*numpy.random.random())

                        def set_rand(num1,num2,val):
                            al = numpy.random.random()
                            x = val[num1]*al+val[num2]*(1-al)
                            x = x + (numpy.random.random()-0.5)*3
                            x = int(x)
                            if x<0: x = 0
                            if x>=len(self.rates): x = len(self.rates)-1
                            return x

                        entr_num  = self.entr_num
                        learn_num = self.learn_num
                        self.entr_num = set_rand(num1,num2, self.entr_nums)
                        self.learn_num = set_rand(num1,num2, self.learn_nums)

                        if self.max_rewards[maxi] > rewm*1.1:
                            for v in range(len(self.entr_nums)):
                                self.entr_nums[v] = self.entr_nums[maxi]
                                self.learn_nums[v] = self.learn_nums[maxi]






                        self.learn_rate = self.rates[self.learn_num]
                        self.c_entr = self.rates[self.entr_num]

                        self.max_rewards[mini] = rewm
                        self.models_time[mini] = 0
                        self.models[mini].set_weights(self.model.get_weights())
                        self.models_time = self.models_time+1.0
                        self.cur_dif_step = 0



                    if(self.cur_step[i]>=self.NSTEP):
                        rewm = self.all_rewards[i].mean()
                        self.rewards_step[i].append(rewm)
                        self.cur_step[i] = 0



                if((i==0) and self.show_on):
                    self.show(i)

        self.learn_all()
        sess.graph.finalize()

        if self.index%10==0:

            print(f"step is {self.index} lr {self.learn_rate} c_entr {self.c_entr}  nums {self.learn_nums} {self.entr_nums}")
            print(f"rewards sum is {self.all_rewards.sum(axis=1).mean()}  rewards mean  {self.all_rewards.mean(axis=1).mean()}")
            rew = 0
            for i in range(self.N):
                rew = rew + self.rewards_step[i][-1]
            rew = rew / self.N


            print(f"reward step {rew}  max_rew {self.max_rewards} cur_dif_step {self.cur_dif_step}  all_rewm {self.all_dif_rewards.mean()}")
            print(f"policy  {self.policies[0,-1]} value {self.values[0,-1]} time {self.models_time}")
            #res1 = self.get_value_res(self.states[0,0:10]).numpy().flatten()
            #res2 = self.get_value_res(self.states[0,-10:]).numpy().flatten()
            #print(f"valuef {res1}")
            #print(f"valuef {res2}")
            if(self.index%500==0 and self.index>500):
                self.model.save_weights("./out/name2.h5")

        self.index = self.index +  1

        return

    def show(self, i):
        cv2.imshow(self.nwin,self.envs.get_cur_state(i)[:,:,0:3])
        if cv2.waitKey(1)==27:
            print("27")
        return



N = 16
envs = multi_environment(N)
ppo1 = ppo(envs)
for i in range(100000000):
    ppo1.step()
