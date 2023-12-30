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
from tensorflow.keras.layers import RepeatVector, Subtract, MaxPooling2D, AveragePooling2D,AveragePooling1D
import tensorflow as tf
import keras.backend as K

numpy.set_printoptions(precision=4)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)
sess.as_default()




class environment():
    def __init__(self):
        self.env = gym.make("Breakout-ram-v4", render_mode="rgb_array")

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
        reward = reward / 10.0
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

        def rescale(x):

            #m1 = LeakyReLU(1e-10) (x*1/16-)
            #y1 = x#concatenate([x/255, m1/16])
            #y2 = x//16
            #y3 = x&0b00000100//4
            #y4 = x&0b00001000//8
            #y5 = x&0b00010000//16
            #y6 = x&0b00100000//32
            #y7 = x&0b01000000//64
            #y8 = x//128

            #y = concatenate([y1, y2])/16-0.5

            y =  tf.cast(x, tf.float32)/512-0.25
            y = Reshape((512,)) (y)



            return y

        print("Shape")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state,  dtype='uint8')

        lay = rescale(inp1)

        lay = Dense(300, activation = 'relu') (lay)
        lay = Dense(200, activation = 'relu') (lay)
        layv1 = Dense(self.len_act, activation = 'linear') (lay)


        self.nnets = 2
        self.modelq = [[]]*self.nnets
        self.modelq[0] = keras.Model(inputs=inp1, outputs=[layv1])

        tf.keras.utils.plot_model(self.modelq[0], to_file='./out/netq.png', show_shapes=True)
        #for i in range(1,self.nnets):
        #    self.modelq[i] = keras.models.clone_model(self.modelq[0])

        self.modelq[1] = keras.models.clone_model(self.modelq[0])

        print("------------")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state, dtype='uint8' )
        #layc1 = Cropping1D((0,1)) (inp1)
        #layc2 = Cropping1D((1,0)) (inp1)
        #layc = Multiply()([layc1,layc2])
        #layc = concatenate([inp1, layc], axis=1)
        lay_r = rescale(inp1)



        lay = Dense(300, activation = 'relu') (lay_r)
        lay = Dense(250, activation = 'relu') (lay)
        lay = Dense(350, activation = 'relu') (lay)

        lay1 = Dense(100, activation="linear") (lay)
        layp1 = Dense(self.len_act, activation="softmax") (lay1)
        lay2 = Dense(200, activation="linear") (lay)

        layp2 = Dense(self.len_act, activation="softmax") (lay2)



        self.modelp = keras.Model(inputs=inp1, outputs=[layp1, layp2])
        self.model_exp= keras.Model(inputs=inp1, outputs=[lay_r])


        self.targetp = keras.models.clone_model(self.modelp)




        self.targetq = [keras.models.clone_model(self.modelq[i]) for i in range(self.nnets)]

        self.nrewards  = 4
        self.max_rewards = numpy.array([0.0]*self.nrewards)
        self.max_nets = [[]]*self.nrewards
        for i in range(self.nrewards):
            self.max_nets[i] = (keras.models.clone_model(self.modelp),
                                keras.models.clone_model(self.targetq[0]), keras.models.clone_model(self.targetq[1]),
                                keras.models.clone_model(self.modelq[0]), keras.models.clone_model(self.modelq[1]))
        self.modelmp = keras.models.clone_model(self.modelp)



        tf.keras.utils.plot_model(self.targetq[0], to_file='./out/nettq1.png', show_shapes=True)
        tf.keras.utils.plot_model(self.modelp, to_file='./out/netp.png', show_shapes=True)


        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=0.0004)

        self.alphav = tf.Variable(0.0031)
        p = 1/self.len_act
        self.entrmax = tf.cast(- tf.math.log(p), tf.float32)
        print(self.entrmax)

        self.max_alpha = 0.01
        self.al_value = 1.0


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
        inp = l_state
        out = self.modelp(inp , training = False)
        #val = self.model_exp(inp , training = False)
        #tf.print(val)

        #out = tf.clip_by_value(out,0.001,0.992)
        return out

    @tf.function
    def get_value_res(self,l_state):

        val = self.modelq[0](l_state/255.0, training = False)[1]
        return val

    def get_net_act(self,l_state):


        out = self.get_net_res(numpy.array([l_state]))
        indv = int(random.random()*1.9999999)

        out1 = out[indv][0].numpy()
        #if random.random()>0.999:
        #    out1[:] = 1.0/self.len_act
        fl = out1 < 0.0001
        if(fl.any() and random.random()>0.9):
            index = int(self.len_act*random.random()*0.99999)
        else:
            index = numpy.random.choice(self.len_act, 1, p = out1) [0]


        return index, out1



    @tf.function
    def train_q1(self, inp, inp_next, actn, rew, dones):
        with tf.GradientTape(persistent=True) as tape1:
            inp1 = inp
            inp_next1 = inp_next
            qv = []
            targ = []
            tqv = []
            for i in range(2):
                qv.append(self.modelq[i](inp1, training = True))
                targ.append(self.targetq[i](inp_next1, training = True))
                tqv.append(self.targetq[i](inp1, training = True))


            y_pi = self.modelp(inp_next1, training = True)

            y_pi = tf.clip_by_value(y_pi,1e-15,0.99999999999999999)
            logpi = tf.math.log(y_pi)

            q = []
            qt = []


            for i in range(2):
                q.append(tf.gather_nd(batch_dims=1,params = qv[i],indices  = actn))
                qt.append( tf.gather_nd(batch_dims=1,params = tqv[i],indices  = actn))


            minq1 = tf.minimum(targ[0] , targ[1])

            dift1 = tf.reduce_sum((minq1-tf.stop_gradient(self.alphav)*logpi[0])*y_pi[0], axis=-1)
            dift2 = tf.reduce_sum((minq1-tf.stop_gradient(self.alphav)*logpi[1])*y_pi[1], axis=-1)
            qvt = []

            qvt.append(tf.stop_gradient(rew+self.gamma*dift1*(1-dones)))
            qvt.append(tf.stop_gradient(rew+self.gamma*dift2*(1-dones)))


            dif = []

            for i in range(2):
               dif1a = tf.math.square(q[i]-qvt[i])
               #dif1b = tf.math.square(qt[i]+tf.clip_by_value(q[i]-qt[i],-0.2,0.2)-qvt[i])
               #dif.append(tf.maximum(dif1a,dif1b))
               dif.append(dif1a)

            lossq = tf.reduce_mean(dif[0]+dif[1])


            trainable_varsa = self.modelq[0].trainable_variables+self.modelq[1].trainable_variables
            #lossw = tf.convert_to_tensor(tf.reduce_mean(tf.math.square(trainable_varsa[0])))
            #lossw = tf.reduce_mean(lossw)
            lossq = lossq # + lossw

        gradsa = tape1.gradient(lossq, trainable_varsa)

        #divkb = tf.convert_to_tensor([tf.reduce_max(tf.math.abs(grad)) for grad in gradsa])
        #divkb = tf.reduce_max(divkb)
        #if(divkb>9e-3):
        #    gradsa = [grad*0.1 for grad in gradsa]
        #else:
        #    gradsa = [grad for grad in gradsa]



        self.optimizer1.apply_gradients(zip(gradsa, trainable_varsa))
        return lossq




    @tf.function
    def train_actor1(self, inp):
        with tf.GradientTape() as tape2:
            inp1 = inp

            qv = []

            for i in range(2):
                qv.append(self.modelq[i](inp1, training = True))

            y_pii = self.modelp(inp1, training = True)
            #y_piim = self.modelmp(inp1, training = True)
            y_pii = tf.clip_by_value(y_pii,1e-15,0.99999999999999)

            #rst1 = tf.reduce_mean(tf.reduce_sum(y_piim[0]*tf.math.log(y_piim[0]/y_pii[0]), axis=-1))
            #rst2 = tf.reduce_mean(tf.reduce_sum(y_piim[1]*tf.math.log(y_piim[1]/y_pii[1]), axis=-1))


            logpi = tf.math.log(y_pii)
            entr = - tf.reduce_mean(tf.reduce_sum(y_pii[0]*logpi[0], axis=-1))
            #divkb = tf.reduce_mean(tf.reduce_sum(y_pii[0]*(logpi[0]-logpi[1]) + y_pii[1]*(logpi[1]-logpi[0]), axis=-1))

            #ypi_border1 = tf.reduce_mean(tf.math.exp(-y_pii[0]/0.00005))
            #ypi_border2 = tf.reduce_mean(tf.math.exp(-y_pii[1]/0.00008))

            #ypi_border1 = tf.reduce_mean(tf.nn.relu(0.00005-y_pii[0]))*1000
            #ypi_border2 = tf.reduce_mean(tf.nn.relu(0.00007-y_pii[1]))*1000


            minq1 = tf.minimum(qv[0], qv[1])

            diflm1 = tf.reduce_sum(y_pii[0]*(tf.stop_gradient(self.alphav)*logpi[0] - minq1),axis=-1)
            diflm2 = tf.reduce_sum(y_pii[1]*(tf.stop_gradient(self.alphav)*logpi[1] - minq1),axis=-1)

            #dst = tf.reduce_mean(tf.math.abs(y_pii1[0] - y_pii1[1]))*0.001

            dm = tf.reduce_mean(diflm1+diflm2) #+ tf.math.exp(-divkb/0.0005)
            lossp = dm  #+ ypi_border1#+ tf.nn.relu(rst1-0.2)+tf.nn.relu(rst2-0.2)#+ (la2c1+la2c2)*0.001#+ tf.square(0.01-dify)*1000.0#+kb*0.0001 #+ tf.nn.relu(self.entrmax*0.01-entr)*10 #+ corrdisp*maxq*0.05 #+ tf.reduce_mean(val)*maxq*0.02 #+ tf.reduce_mean(entr)*maxq*0.0005

            trainable_vars2 = self.modelp.trainable_variables
            #lossw = tf.convert_to_tensor(tf.reduce_mean(tf.math.square(trainable_vars2[0])) )
            #lossw = tf.reduce_mean(lossw)
            lossp = lossp  #+ 0.5*lossw


        grads2 = tape2.gradient(lossp, trainable_vars2)

        self.optimizer2.apply_gradients(zip(grads2, trainable_vars2))

        return  lossp, 0.0, entr





    @tf.function
    def target_train(self):
        tau = [0.005,0.0050, 0.005]
        for i in range(2):
            target_weights = self.targetq[i].trainable_variables
            weights = self.modelq[i].trainable_variables
            for (a, b) in zip(target_weights, weights):
                a.assign(a * (1-tau[i]) + b*tau[i])
        return

    @tf.function
    def train_dif_ev(self, inum1, inum2):
        tau = [0.5,0.5]
        tf.print(inum1, " ",inum2)
        al = random.random()*0.5
        for i in range(2):
            target_weights = self.targetq[i].trainable_variables
            weights1 = self.max_nets[inum1][int(i+1)].trainable_variables
            weights2 = self.max_nets[inum2][int(i+1)].trainable_variables

            for (a, m1, m2) in zip(target_weights, weights1, weights2):
                al1 = random.random()*0.01
                a.assign(a+(m2-m1)*(al+al1))

        for i in range(2):
            target_weights = self.modelq[i].trainable_variables
            weights1 = self.max_nets[inum1][int(i+3)].trainable_variables
            weights2 = self.max_nets[inum2][int(i+3)].trainable_variables
            for (a, m1,m2) in zip(target_weights, weights1, weights2):
                al1 = random.random()*0.00005
                a.assign(a+(m2-m1)*(al+al1))


        #target_weights = self.modelp.trainable_variables
        #weights1 = self.max_nets[inum1][0].trainable_variables
        #weights2 = self.max_nets[inum2][0].trainable_variables
        #for (a, m1,m2) in zip(target_weights, weights1, weights2):
        #    al1 = random.random()*0.0005
        #    a.assign(a+(m2-m1)*(al+al1))

        tf.print("make dif ev")
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
                #inum = int(self.nrewards*random.random()*0.9999)
                inum = self.max_rewards.argmin()
                mean = self.max_rewards.max()


                #self.alphav = tf.Variable(0.003+self.cur_reward100*0.0005)
                #if self.cur_reward100 > self.max_rewards[inum]:


                self.max_rewards[inum] = self.cur_reward100
                self.max_nets[inum][0].set_weights(self.modelp.get_weights())
                self.max_nets[inum][1].set_weights(self.targetq[0].get_weights())
                self.max_nets[inum][2].set_weights(self.targetq[1].get_weights())

                self.max_nets[inum][3].set_weights(self.modelq[0].get_weights())
                self.max_nets[inum][4].set_weights(self.modelq[1].get_weights())




                val = numpy.arange(self.nrewards)
                val = numpy.delete(val, inum)
                numpy.random.shuffle(val)
                if self.max_rewards[val[0]]>self.max_rewards[val[1]]:
                    vl = val[0]
                    val[0] = val[1]
                    val[1] = vl

                #if(self.cur_reward100 < mean*0.9):
                    #self.train_dif_ev(int(val[0]),int(val[1]))








        if self.flag:
            self.show()

        if self.index>self.T*4 and self.index%64==0:
            #if self.index>12e+6:
            #    vali = 12e+6
            #else:
            #    vali = self.index
            #self.alphav = tf.Variable(vali*1.5e-9+0.001)
            lossq, lossp, qv, qvt = self.learn_all()
            if(random.random()>0.99):
                self.fl_mod = not self.fl_mod




            if(self.index%4000==0 and self.buf_index>self.T):
                print(f"index {self.index} {self.alphav.numpy():.{2}e} maxrew {self.max_rewards} lossq {lossq:.{2}e}  lossp {lossp:.{2}e} qv {qv:.{2}e}  qvt {qvt:.{2}e} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100:.{3}f} ")
                self.cur_reward = 0

                self.show()
                if(self.index%32000==0):
                    plt.plot(self.xindex,self.rewardy)
                    plt.savefig("./out/figure_rew7.png")
                    plt.close()
                    df = pandas.DataFrame({'x': self.xindex, 'y': self.rewardy})
                    df.to_excel('./out/file_name7.xlsx', index=False)


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
