# MIT License
# Copyright (c) 2023 saysaysx
import random
import numpy
import time

from random import choices
import tensorflow as tf
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
import pandas

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization, Dropout, Conv2D, Reshape, Flatten
from tensorflow.keras.layers import MaxPooling1D, Permute, Conv1D, LSTM, LeakyReLU, Cropping1D, Multiply, Softmax, GaussianNoise
from tensorflow.keras.layers import RepeatVector, Subtract, Embedding
import tensorflow as tf
from tensorflow.keras import regularizers
import keras.backend as K

numpy.set_printoptions(precision=4)

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

        #self.env = wr.AtariWrapper(self.env,frame_skip=1, terminal_on_life_loss=False)
        #self.env = wr.NoopResetEnv(self.env)
        #self.env = wr.FireResetEnv(self.env)

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




class sac:
    def __init__(self,env):
        self.env = env
        self.index = 0

        self.max_size = self.index
        self.flag = False

        self.shape_state = env.get_shape_state()
        self.len_act = env.get_len_acts()
        self.gamma = 0.999
        self.alpha = tf.Variable(1.0)


        self.max_t = 10
        self.start_time = time.time()

        self.T = 512
        self.n_buffer = 80000
        self.buf_index  = 0
        self.flag_buf = False
        self.indT = tf.range(self.T)



        self.rews = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.acts = numpy.zeros((self.n_buffer, 1),dtype=numpy.float32)
        self.policies = numpy.zeros((self.n_buffer, self.len_act),dtype=numpy.float32)
        self.qval = numpy.zeros((self.n_buffer, self.len_act),dtype=numpy.float32)
        self.values = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.previous_states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.dones = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.vars = [i for i in range(self.len_act)]


        def rescale(x):

            y =  tf.cast(x, tf.float32)/255-0.5
            y = Reshape((512,)) (y)
            return y

        print("Shape")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state,  dtype='uint8')

        lay = rescale(inp1)

        lay = Dense(350, activation = 'relu') (lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = Dense(150, activation = 'relu') (lay)
        lay = Dense(150, activation = 'relu') (lay)
        layv1 = Dense(self.len_act, activation = 'linear') (lay)


        self.nnets = 2
        self.modelq = [[]]*self.nnets
        self.modelq[0] = keras.Model(inputs=inp1, outputs=[layv1])

        tf.keras.utils.plot_model(self.modelq[0], to_file='./out/netq.png', show_shapes=True)

        for i in range(1,self.nnets):
            self.modelq[i] = keras.models.clone_model(self.modelq[0])

        print("------------")
        print(self.shape_state)
        self.step_s = steps(2)

        inp1 = Input(shape = self.shape_state, dtype='uint8' )

        lay = rescale(inp1)
        coefv = 1e-3*self.len_act

        lay = Dense(350, activation = 'relu') (lay)
        lay = Dense(250, activation = 'relu') (lay)
        lay = Dense(150, activation = 'relu') (lay)
        lay = Dense(150, activation = 'relu') (lay)
        layp1 = Dense(self.len_act, activation="softmax") (lay)
        self.modelp = keras.Model(inputs=inp1, outputs=layp1)


        self.targetq = [keras.models.clone_model(self.modelq[i]) for i in range(self.nnets)]

        self.nrewards  = 3
        self.max_rewards = numpy.array([0.0]*self.nrewards)
        self.max_nets = [[]]*self.nrewards
        for i in range(self.nrewards):
            self.max_nets[i] = (keras.models.clone_model(self.modelp),
                                keras.models.clone_model(self.targetq[0]), keras.models.clone_model(self.targetq[1]),
                                keras.models.clone_model(self.modelq[0]), keras.models.clone_model(self.modelq[1]))

        self.modelqmax = [keras.models.clone_model(self.targetq[0]),keras.models.clone_model(self.targetq[1])]

        tf.keras.utils.plot_model(self.targetq[0], to_file='./out/nettq1.png', show_shapes=True)
        tf.keras.utils.plot_model(self.modelp, to_file='./out/netp.png', show_shapes=True)

        self.hash_len = 32
        inp1 = Input(shape = self.shape_state, dtype='uint8' )
        lay = rescale(inp1)

        inpt = Input(shape=(1,))
        emb = Embedding(1,128) (inpt)
        emb = Flatten()(emb)
        lay = concatenate([lay,emb])
        lay = Dense(200, activation = 'relu') (lay)
        lay = Dense(100, activation = 'relu') (lay)
        layh = Dense(self.hash_len, activation="linear") (lay)
        self.hash_model = keras.Model(inputs=[inp1, inpt], outputs=layh)
        self.rnd_model = keras.models.clone_model(self.hash_model)



        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer4 = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.alphav = tf.Variable(0.02)
        self.border = tf.Variable(0.1)
        self.std_rnd = tf.Variable(1.0)
        self.mean_rnd = tf.Variable(0.0)
        self.bettav = tf.Variable(0.0)

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
        cv2.namedWindow(self.nwin)
        cv2.setMouseCallback(self.nwin,self.capture_event)
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
        val = self.modelq[0](inp , training = False)

        return out, val

    @tf.function
    def get_value_res(self,l_state):

        val = self.modelq[0](l_state, training = False)[1]
        return val

    def get_net_act(self,l_state):
        out, val = self.get_net_res(numpy.array([l_state]))
        out1 = out[0].numpy()
        val1 = val[0].numpy()
        index = numpy.random.choice(self.len_act, 1, p = out1) [0]

        return index, out1, val1


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
                #tqv.append(self.targetq[i](inp, training = True))

            y_pi = self.modelp(inp_next, training = True)
            y_pii = self.modelp(inp, training = True)
            y_pii = tf.clip_by_value(y_pii,1e-10,1.0)
            log = tf.math.log(y_pi)

            logpi = tf.math.log(y_pii)
            pol = tf.clip_by_value(pol,1e-10,1.0)
            logpol = tf.math.log(pol)
            q, qt = [], []

            for i in range(self.nnets):
                q.append(tf.gather_nd(batch_dims=1,params = qv[i],indices  = actn))
                #qt.append( tf.gather_nd(batch_dims=1,params = tqv[i],indices  = actn))

            minq  = tf.minimum(targ[0] , targ[1])



            nentr = self.alphav*logpi*y_pii
            kl = y_pii*(logpi-logpol)


            #hash = self.hash_model([inp,actn], training = True)

            #rndhash = self.rnd_model([inp,actn], training = True)
            #difhash = tf.reduce_mean(tf.math.square(hash-rndhash), axis=-1)
            #losshash = tf.reduce_mean(difhash)

            #acts = tf.random.categorical(log,1)
            #acts = tf.math.argmax(minq,axis=-1)[:,None]

            #minq1 = tf.gather_nd(batch_dims=1,params = minq,indices  = acts)

            #entr  = tf.gather_nd(batch_dims=1,params = nentr,indices  = actn)
            #klg = tf.gather_nd(batch_dims=1,params = kl,indices  = actn)

            minq1  = tf.reduce_sum(y_pi*minq,axis=-1)
            entr = tf.reduce_sum(nentr,axis=-1)
            klg =  tf.reduce_sum(kl,axis=-1)
            klg = tf.nn.relu(klg-0.1)
            # anti exploration
            self.bettav.assign(tf.maximum(tf.reduce_max(rew),self.bettav))
            qvt = rew/self.bettav +self.gamma*minq1*(1-dones) - entr - klg*0.1



            dif = []
            val = []
            dif1a = tf.reduce_mean(tf.math.square(q[0]-qvt))
            dif1b = tf.reduce_mean(tf.math.square(q[1]-qvt))


            dif = [dif1a,dif1b]



            lossq = tf.reduce_mean(tf.convert_to_tensor(dif,tf.float32))
            trainable_varsa = self.modelq[0].trainable_variables + self.modelq[1].trainable_variables
            #trainable_varsb = self.rnd_model.trainable_variables

        gradsa = tape1.gradient(lossq, trainable_varsa)
        self.grad_accum = [acc.assign(acc+g) for acc, g in zip(self.grad_accum, gradsa)]


        #gradsa = [grad for grad in gradsa]
        #if self.bettav>3:
        #    self.optimizer1.apply_gradients(zip(self.grad_accum, trainable_varsa))
        #    self.grad_accum = [grad*0.0 for grad in gradsa]
        #    self.bettav.assign(0.0)

        #gradsb = tape1.gradient(losshash, trainable_varsb)
        #self.optimizer3.apply_gradients(zip(gradsb, trainable_varsb))


        return self.bettav

    @tf.function
    def apply_grads(self):
        trainable_varsa = self.modelq[0].trainable_variables + self.modelq[1].trainable_variables
        grads = [grad/2.0 for grad in self.grad_accum]
        self.optimizer1.apply_gradients(zip(grads, trainable_varsa))
        for acc in self.grad_accum:
            acc.assign(tf.zeros_like(acc))





    @tf.function
    def train_actor1(self, inp, pol, val):
        with tf.GradientTape(persistent=True) as tape2:

            qv, qvt = [], []
            for i in range(self.nnets):
                qv.append(self.modelq[i](inp, training = True))
                qvt.append(val)
            y_pii = self.modelp(inp, training = True)
            y_pii = tf.clip_by_value(y_pii,1e-20,1.0)
            logpi = tf.math.log(y_pii)
            #pol = tf.clip_by_value(pol,1e-10,1.0)
            #logpol = tf.math.log(pol)


            entr = - tf.reduce_mean(tf.reduce_sum(y_pii*logpi, axis=-1))


            minq = tf.minimum(qv[0],qv[1])
            #minq1 = minq/self.alphav
            #qe = tf.math.exp(minq1 - tf.reduce_max(minq1,axis=-1)[:,None])
            #qsum1 = tf.reduce_sum(qe,axis=-1)
            #qe = qe / qsum1[:,None]
            #qe = tf.clip_by_value(qe,1e-10,1.0)
            #logqe = tf.math.log(qe)

            #f1 = tf.math.cumsum(y_pii,axis=-1)
            #f2 = tf.math.cumsum(qe,axis=-1)
            #g1 = tf.math.cumsum(y_pii,axis=-1, reverse=True)
            #g2 = tf.math.cumsum(qe,axis=-1, reverse=True)
            #dif1 = tf.math.square(f2-f1) + tf.math.square(g2-g1)
            #difabs = tf.stop_gradient(tf.abs(dif1))

            dif1 = tf.reduce_sum(y_pii*(logpi*self.alphav-minq),axis=-1)

            #dif1 = tf.reduce_sum(y_pii*(logpi-logqe1),axis=-1)*tf.reduce_sum(qe1*(logqe1-logpi),axis=-1)
            #dif1 = tf.reduce_max(tf.math.abs(f2-f1),axis=-1)
            lossp = tf.reduce_mean(dif1)
            #lossbt = - tf.reduce_mean(dif2)

            trainable_vars2 = self.modelp.trainable_variables


        grads2 = tape2.gradient(lossp, trainable_vars2)
        self.optimizer2.apply_gradients(zip(grads2, trainable_vars2))
        #gradsb = tape2.gradient(lossbt, [self.bettav])
        #self.optimizer4.apply_gradients(zip(gradsb, [self.bettav]))


        return  lossp, tf.reduce_mean(entr) , lossp


    @tf.function
    def target_train(self):
        tau = [0.006,0.006]
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
        val = tf.cast(self.qval[indices] ,tf.float32)

        lossq = self.train_q1(inp,inp_next,acts, rews, dones, pol)
        if self.step_s():
            self.apply_grads()

        lossp, entr, dif = self.train_actor1(inp, pol, val)
        time.sleep(0.01)

        self.target_train()
        return lossq, lossp, entr, dif

    def add(self, reward,done,prev_state,state,act, pol, qval):
        i = self.buf_index
        self.rews[i] = reward
        self.dones[i] = done
        self.states[i] = state
        self.previous_states[i] = prev_state
        self.acts[i] = act
        self.policies[i] = pol
        self.qval[i] = qval
        self.buf_index = self.buf_index+1
        if self.buf_index>=self.n_buffer:
            self.buf_index = 0
            self.flag_buf = True

        return


    def step(self):

        prev_st  = self.env.state()
        act, pol, val = self.get_net_act(prev_st)

        state, reward, done  = self.env.get_state(act)

        self.add(reward,done,prev_st,state,act, pol, val)

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
                    self.bettav.assign(self.max_reward)
                    self.modelqmax[0].set_weights(self.targetq[0].get_weights())
                    self.modelqmax[1].set_weights(self.targetq[1].get_weights())


                inum = self.max_rewards.argmin()


                self.max_rewards[inum] = self.cur_reward100
                self.max_nets[inum][0].set_weights(self.modelp.get_weights())
                self.max_nets[inum][1].set_weights(self.targetq[0].get_weights())
                self.max_nets[inum][2].set_weights(self.targetq[1].get_weights())

                self.max_nets[inum][3].set_weights(self.modelq[0].get_weights())
                self.max_nets[inum][4].set_weights(self.modelq[1].get_weights())



                mean = self.max_rewards.max()
                #self.alpha = tf.Variable(1.2/((mean*0.02+1.0)))
                #self.alphav = tf.Variable(0.01/((mean*0.02+1.0)))
                #self.border = tf.Variable(0.15*((mean*0.05+1.0)))

        if self.flag:
            self.show()

        if self.index>self.T*4 and self.index%64==0:

            lossq, lossp, entr, dif = self.learn_all()


            if(self.index%4000==0 and self.buf_index>self.T):
                print(f"index {self.index} {self.alphav.numpy():.{2}e} maxrew {self.max_rewards} lossq {lossq:.{2}e}  lossp {lossp:.{2}e} entr {entr:.{2}e}  dif {dif:.{2}e} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100:.{3}f} ")
                self.cur_reward = 0

                self.show()
                if(self.index%32000==0):
                    plt.plot(self.xindex,self.rewardy)
                    plt.savefig("./out/figure_rew8.png")
                    plt.close()
                    df = pandas.DataFrame({'x': self.xindex, 'y': self.rewardy})
                    df.to_excel('./out/file_name8.xlsx', index=False)


        self.index = self.index+1
        return

    def show(self):
        cv2.imshow(self.nwin,self.env.get_image())
        if cv2.waitKey(1)==27:
            print("27")
            self.flag = not self.flag
        return




env = environment()
sac1 = sac(env)
for i in range(100000000):
    sac1.step()
