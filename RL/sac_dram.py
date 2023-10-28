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
from tensorflow.keras.layers import MaxPooling1D, Permute, Conv1D, LSTM
import tensorflow as tf
import keras.backend as K

numpy.set_printoptions(precision=4)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)
sess.as_default()




class environment():
    def __init__(self):
        self.env = gym.make("MsPacman-ram-v4", render_mode="rgb_array")

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
        reward = reward / 100.0
        next_observation = numpy.array(next_observation)


        shx = (next_observation).transpose([0,1]).shape
        self.field = numpy.reshape(next_observation.transpose([0,1]),[shx[0],shx[1]])


        self.reward = self.reward+reward
        self.index = self.index + 1
        #if(self.index>1000):
        #    self.index = 0
        #    done = True

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
        self.alpha = 0.5


        self.max_t = 10
        self.start_time = time.time()

        self.T = 512
        self.n_buffer = 80000
        self.buf_index  = 0
        self.flag_buf = False



        self.rews = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.acts = numpy.zeros((self.n_buffer, 1),dtype=numpy.float32)
        self.policies = numpy.zeros((self.n_buffer, self.len_act),dtype=numpy.float32)
        self.values = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.previous_states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.dones = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.vars = [i for i in range(self.len_act)]

        def dconv(filt,size,x, strides=2):
            y = keras.layers.UpSampling2D(size=strides)(x)
            y = keras.layers.Conv2D(filters = filt,kernel_size=size,activation = 'tanh',padding='same') (y)
            return y





        print("Shape")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state)
        lay = Flatten() (inp1)
        lay = Dense(300, activation="relu") (lay)
        lay = Dense(200, activation="relu") (lay)

        lay = Dense(60, activation="relu") (lay)

        layv = Dense(self.len_act, activation = 'linear') (lay)


        self.nnets = 4
        self.modelq = [[]]*self.nnets
        self.modelq[0] = keras.Model(inputs=inp1, outputs=[layv])
        for i in range(1,self.nnets):
            self.modelq[i] = keras.models.clone_model(self.modelq[0])



        print("------------")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state)
        lay = Flatten() (inp1)
        lay = Dense(300, activation="relu") (lay)
        lay = Dense(200, activation="relu") (lay)

        lay = Dense(60, activation="relu") (lay)

        layp = Dense(self.len_act, activation="softmax") (lay)

        self.modelp = keras.Model(inputs=inp1, outputs=[layp])



        self.targetp = keras.models.clone_model(self.modelp)




        self.targetq = [keras.models.clone_model(self.modelq[0]) for i in range(self.nnets)]

        tf.keras.utils.plot_model(self.modelq[0], to_file='./out/netq.png', show_shapes=True)
        tf.keras.utils.plot_model(self.targetq[0], to_file='./out/nettq1.png', show_shapes=True)
        tf.keras.utils.plot_model(self.modelp, to_file='./out/netp.png', show_shapes=True)


        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=0.0004)

        self.alphav = tf.Variable(0.003)
        p = 1/self.len_act
        self.entrmax = tf.cast(- tf.math.log(p), tf.float32)
        print(self.entrmax)

        self.max_alpha = 0.01


        self.cur_reward = 0.0
        self.max_reward = 1.0
        self.rand_true = False

        self.cur_reward100 = 0.0
        self.cur_reward10 = 0.0
        self.num_games = 0
        self.flag = True
        self.fl_mod = True
        self.num = 0



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
        inp = tf.cast(l_state,tf.float32)/255.0 - 0.5
        out = self.modelp(inp , training = False)
        #out = tf.clip_by_value(out,0.001,0.992)

        return out

    @tf.function
    def get_value_res(self,l_state):

        val = self.modelq[0](l_state/255.0, training = False)[1]
        return val

    def get_net_act(self,l_state):


        out = self.get_net_res(numpy.array([l_state]))
        out1 = out[0].numpy()
        #if random.random()>0.999:
        #    out1[:] = 1.0/self.len_act


        vars = [i for i in range(self.len_act)]
        index = numpy.array(choices(vars, out1)[0])
        return index, out1



    @tf.function
    def train_q1(self, inp, inp_next, actn, rew, dones):
        with tf.GradientTape(persistent=True) as tape1:
            inp1 = inp / 255.0 - 0.5
            inp_next1 = inp_next / 255.0 - 0.5
            qv = []
            targ = []
            #tqv = []
            for i in range(2):
                qv.append(self.modelq[i](inp1, training = True))
                targ.append(self.targetq[i](inp_next1, training = True))
                #tqv.append(self.targetq[i](inp1, training = True))


            minq1 = (targ[0]+targ[1])*0.5
            #minq2 = tf.minimum(targ[2],targ[3])
            minq = minq1#(minq1+minq2)*0.5


            targ = minq#(targ[0]+targ[1]+targ[2]+targ[3])*0.25#tf.minimum(targ1,targ2)#
            y_pi = self.modelp(inp_next1, training = True)
            y_pi = tf.clip_by_value(y_pi,1e-15,0.99999999999999999)
            logpi = tf.math.log(y_pi)

            q = []
            #qt = []
            for i in range(2):
                q.append(tf.gather_nd(batch_dims=1,params = qv[i],indices  = actn))
                #qt.append( tf.gather_nd(batch_dims=1,params = tqv[i],indices  = actn))


            dift = tf.reduce_sum((targ-tf.stop_gradient(self.alphav)*logpi)*y_pi, axis=-1)
            qvt =  tf.stop_gradient(rew+self.gamma*dift*(1-dones))
            dif = []

            for i in range(2):
               dif.append(tf.math.square(q[i]-qvt))

            lossq = tf.reduce_mean(dif[0]+dif[1])


            trainable_varsa = self.modelq[0].trainable_variables+self.modelq[1].trainable_variables
        gradsa = tape1.gradient(lossq, trainable_varsa)
        self.optimizer1.apply_gradients(zip(gradsa, trainable_varsa))
        return lossq




    @tf.function
    def train_actor1(self, inp):
        with tf.GradientTape() as tape2:
            inp1 = inp / 255.0 - 0.5

            qv = []

            for i in range(2):
                qv.append(self.modelq[i](inp1, training = True))

            y_pii = self.modelp(inp1, training = True)
            y_pii = tf.clip_by_value(y_pii,1e-15,0.99999999999999)

            logpi = tf.math.log(y_pii)
            entr = - tf.reduce_mean(tf.reduce_sum(y_pii*logpi, axis=-1))
            ypi_border = tf.reduce_mean(tf.math.exp(-y_pii/0.00001))*1e+3#tf.reduce_mean(tf.nn.relu(1e-7 - y_pii))*1e+7
            minq1 = (qv[0]+qv[1])*0.5
            #minq2 = tf.minimum(qv[2],qv[3])
            minq = tf.stop_gradient(minq1)#(minq1+minq2)*0.5
            #minq = tf.stop_gradient((qv[0]+qv[1]+qv[2]+qv[3])*0.25)##t
            diflm = tf.reduce_sum(y_pii*(tf.stop_gradient(self.alphav)*logpi - minq),axis=-1)
            dm = tf.reduce_mean(diflm)


            lossp = dm + ypi_border #+kb*0.0001 #+ tf.nn.relu(self.entrmax*0.01-entr)*10 #+ corrdisp*maxq*0.05 #+ tf.reduce_mean(val)*maxq*0.02 #+ tf.reduce_mean(entr)*maxq*0.0005



            trainable_vars2 = self.modelp.trainable_variables

        grads2 = tape2.gradient(lossp, trainable_vars2)

        self.optimizer2.apply_gradients(zip(grads2, trainable_vars2))

        return  lossp, ypi_border, entr





    @tf.function
    def target_train(self):
        tau = [0.01,0.010,0.01,0.01]
        for i in range(2):
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

        inp_next = tf.cast(self.states[indices] ,tf.float32)
        inp = tf.cast(self.previous_states[indices],tf.float32)

        acts = tf.cast(self.acts[indices] ,tf.int32)
        rews = tf.cast(self.rews[indices] ,tf.float32)
        dones = tf.cast(self.dones[indices] ,tf.float32)


        self.num = int(not bool(self.num))

        lossq = self.train_q1(inp,inp_next,acts, rews, dones)
        lossp, qv, qvt = self.train_actor1(inp)



        #time.sleep(0.05)
        self.target_train()
        return lossq, lossp, qv, qvt

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

                    self.targetp.set_weights(self.modelp.get_weights())


        if self.flag:
            self.show()

        if self.index>self.T*4 and self.index%64==0:
            if self.index>10e+6:
                vali = 10e+6
            else:
                vali = self.index
            self.alphav = tf.Variable(vali*2.0e-9+0.003)
            lossq, lossp, qv, qvt = self.learn_all()
            if(random.random()>0.99):
                self.fl_mod = not self.fl_mod




            if(self.index%4000==0 and self.buf_index>self.T):
                print(f"index {self.index} alphav {self.alphav.numpy():.{3}f} {self.rand_true} lossq {lossq:.{2}e}  lossp {lossp:.{2}e} qv {qv:.{2}f}  qvt {qvt:.{2}f} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100:.{3}f} ")
                self.cur_reward = 0

                self.show()
                if(self.index%32000==0):
                    plt.plot(self.xindex,self.rewardy)
                    plt.savefig("./out/figure_rew6.png")
                    plt.close()
                    df = pandas.DataFrame({'x': self.xindex, 'y': self.rewardy})
                    df.to_excel('./out/file_name6.xlsx', index=False)


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
