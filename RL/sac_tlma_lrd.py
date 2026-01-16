import numpy as np
import tensorflow as tf
import gymnasium as gym

import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


# Убедитесь, что директория вывода существует
os.makedirs('./out', exist_ok=True)

#gym.register(id="Tetris-v0", entry_point="tetris_env:TetrisEnv")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[1 if len(gpus) > 1 else 0], 'GPU')

class Logger:
    def __init__(self, log_file='./out/training.log'):
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write("Training Log Started\n")

    def log(self, message):

        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

class EnvironmentWrapper:
    def __init__(self):
        #self.env = gym.make("Tetris-v0", render_mode="rgb_array")
        self.env = gym.make("Breakout-v4", render_mode="rgb_array")
        self.env = gym.wrappers.GrayscaleObservation(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env, shape=(96, 96))
        self.env = gym.wrappers.FrameStackObservation(self.env, stack_size=4)

        self.n_action = self.env.action_space.n
        self.index = 0
        self.reward = 0.0
        self.env_reset()

    def get_state(self, act):
        self.index += 1
        next_obs, reward, done, _, _ = self.env.step(act)
        self.reward += reward
        self.field = np.array(next_obs)
        return self.state(), reward, done

    def env_reset(self):
        self.index = 0
        self.reward = 0.0
        self.field = np.array(self.env.reset()[0])

    def get_image(self):
        return self.env.render()

    def state(self):
        return self.field

    def get_shape_state(self):
        return self.state().shape

    def get_len_acts(self):
        return self.n_action

# ========================
# УПРОЩЁННЫЙ ReplayBuffer
# ========================
class ReplayBuffer:
    def __init__(self, state_shape, n_actions, buffer_size=40000):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.buffer_size = buffer_size

        # Храним ПОЛНЫЙ переход: (s, a, r, s_next, done, entropy)
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((buffer_size, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.policy = np.zeros((buffer_size, n_actions), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, state, action, reward, next_state, done, policy=[]):
        """Добавляет ПОЛНЫЙ переход."""
        self.states[self.idx] = state
        self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = bool(done)
        self.policy[self.idx] = policy

        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True

    def sample_batch(self, batch_size):
        max_idx = self.buffer_size if self.full else self.idx
        if max_idx == 0:
            return None
        indices = np.random.choice(max_idx, batch_size)#np.random.randint(0, max_idx, size=batch_size)
        return {
            'states': tf.cast(self.states[indices], tf.uint8),
            'next_states': tf.cast(self.next_states[indices], tf.uint8),
            'actions': tf.cast(self.actions[indices], tf.int32),
            'rewards': tf.cast(self.rewards[indices], tf.float32),
            'dones': tf.cast(self.dones[indices], tf.float32),
            'policy': tf.cast(self.policy[indices], tf.float32),
        }



def get_ind_r(x):
    CONST_MINV = 0.0011 / (1 - 0.0011 * len(x))
    x = x + CONST_MINV
    f = np.cumsum(x) / (1 + CONST_MINV * len(x))
    v = np.random.random()
    index = np.digitize(v, f)
    return min(index, len(x) - 1)


def random_shift(x, pad=1):
    """
    Быстрый random shift для [B, T, H, W], uint8.
    Использует эффективные операции tf.image через reshape.
    """
    assert x.dtype == tf.uint8, "Ожидается tf.uint8"

    B, T, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

    # Преобразуем в [B*T, H, W, 1]
    x_flat = tf.reshape(x, [B * T, H, W, 1])

    # Добавляем padding: [B*T, H+2*pad, W+2*pad, 1]
    x_padded = tf.pad(x_flat, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='CONSTANT')

    # Случайный crop обратно до [H, W]
    # tf.image.random_crop работает по последним двум измерениям (H, W)
    x_cropped = tf.image.random_crop(x_padded, size=[B * T, H, W, 1])

    # Возвращаем к [B, T, H, W]
    x_out = tf.reshape(x_cropped, [B, T, H, W])

    return x_out



class SACAgent:
    def __init__(self, env, logger):
        self.env = env
        self.logger = logger
        self.shape_state = env.get_shape_state()
        self.len_act = env.get_len_acts()
        self.start_time = datetime.now()

        self.buffer = ReplayBuffer(
            state_shape=self.shape_state,
            n_actions=self.len_act,
            buffer_size=40000

        )

        self.T = 32
        self.gamma = tf.Variable(0.99, trainable=False)
        self.index = 0
        self.cur_reward = 0.0
        self.num_games = 0
        self.cur_reward10 = 0.0
        self.cur_reward100 = 0.0
        self.max_reward = 1.0
        self.last_reward = 1.0
        self.num_learn = 1

        self.n_heads = 2
        self.n_outs = 2
        self.i_out = 0
        self.mean_index = 400
        self.tau = 0.99
        self.tau_v = tf.Variable(0.99, trainable=False)
        self.mm_old = tf.Variable(0.0, trainable=False)
        self.mm_new = tf.Variable(0.0, trainable=False)
        self.learn_target = 1
        self.log_nact = np.log(self.env.n_action)

        self.nwin = "Main"
        self.flag = False
        #cv2.namedWindow(self.nwin)
        #cv2.setMouseCallback(self.nwin, self.capture_event)

        from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, UpSampling2D, Concatenate, BatchNormalization, Lambda
        from tensorflow.keras import Model

        from keras.layers import Lambda
        from tensorflow.keras.utils import register_keras_serializable
        @register_keras_serializable()
        def rescale(x):
            y = tf.cast(x, tf.float32) / 256.0
            return tf.transpose(y, perm=[0, 2, 3, 1])


        # Encoder
        inp = Input(shape=self.shape_state, dtype='uint8', name='inp')
        x = Lambda(rescale)(inp)
        x = Conv2D(32, (8,8), strides=4, padding='same', activation='relu')(x)
        x = Conv2D(64, (4,4), strides=2, padding='same', activation='relu')(x)
        x = Conv2D(64, (3,3), strides=2, padding='same', activation='relu', name='spatial_features')(x)
        spatial_features = x  # (B, 6, 6, 64)
        flat = Flatten()(spatial_features)


        encoder_out = Dense(512, activation='relu', name = 'enc_out')(flat)

        self.encoder = Model(inputs=inp, outputs=encoder_out)



        # Q-heads
        q_out = []
        for j in range(self.n_outs):
            q_hidden = Dense(512, activation='relu', name=f'q_hidden_{j}')(self.encoder.get_layer("enc_out").output)
            q_out.append(Dense(self.len_act, activation='linear', name=f'q_head_{j}')(q_hidden))


        self.q_heads = [Model(inputs=self.encoder.get_layer("inp").output, outputs=q_out)]
        for i in range(1, self.n_heads):
            self.q_heads.append(tf.keras.models.clone_model(self.q_heads[0]))

        self.target_q_heads = []
        for i in range(self.n_heads):
            self.target_q_heads.append(tf.keras.models.clone_model(self.q_heads[0]))


        self.policy_encoder = tf.keras.models.clone_model(self.encoder)
        # Policy head
        pi_out = []
        for i in range(self.n_outs):
            pi_out.append(Dense(self.len_act, activation='softmax')(self.policy_encoder.get_layer("enc_out").output))

        self.policy_net = Model(inputs=self.policy_encoder.get_layer("inp").output, outputs=pi_out)

        self.target_pi = tf.keras.models.clone_model(self.policy_net)

        # Optimizers

        self.optimizer_pi = tf.keras.optimizers.Adam(2.0e-4)
        self.optimizer = [tf.keras.optimizers.Adam(learning_rate=0.0002) for i in range(self.n_heads)]

        self.opt = [self.train_q(self.optimizer[i], self.q_heads[i]) for i in range(self.n_heads)]


        # Logging
        self.xindex = []
        self.rewardy = []
        self.cur_minq = tf.Variable(0.0)




    def capture_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.flag = not self.flag



    @tf.function
    def get_net_res(self,l_state,iout):
        inp = l_state
        out = self.policy_net(inp, training = False)[iout]
        return out

    def get_action(self, l_state):

        out = self.get_net_res(np.array([l_state]), self.i_out)[0].numpy()
        index = get_ind_r(out)
        return index, out

    def train_q(self, optimizer, model):
        @tf.function
        def train(states, actions, rewards, next_states, dones):

            aug_states = states

            aug_next_states = next_states

            with tf.GradientTape(persistent=True) as tape:

                q = model(aug_states, training=True)
                pi_next = tf.reduce_mean(self.target_pi(aug_next_states, training=True),axis=0)


                qt = []
                for i in range(self.n_heads):
                    qt.append(tf.convert_to_tensor(self.target_q_heads[i](aug_next_states, training=True)))

                qt = tf.reduce_mean(tf.stack(qt),axis=0)

                lognext = tf.math.log(pi_next+1e-9)
                entr = -tf.reduce_sum(pi_next * lognext, axis=-1)/self.log_nact

                loss_q = 0
                maxx = 0.0
                for i in range(self.n_outs):
                    #maxx = tf.reduce_max(qt[i],axis=-1)
                    #qe = tf.exp(tf.clip_by_value((qt[i]-maxx[:,None])/0.1,-6.5,0.0))

                    #qsum = tf.reduce_sum(qe,axis=-1)
                    #qe = qe / (qsum[:,None])
                    qe = pi_next[i]
                    log = - tf.math.log(qe+1e-12)
                    N = tf.cast(self.len_act,tf.float32)
                    entr = tf.reduce_sum(log*qe,axis=-1)/tf.math.log(N)

                    v_target = tf.reduce_sum(qe*qt[i], axis=-1) + entr*0.0002
                    y = rewards + self.gamma * (1 - dones) * v_target
                    q_a = tf.gather_nd(batch_dims=1,params = q[i],indices  = actions)
                    loss_q += tf.reduce_mean(tf.square(q_a - y))
                    maxx += tf.reduce_max(v_target)




                loss = loss_q/self.n_outs
                maxx = maxx/self.n_outs
                self.mm_old.assign(self.mm_old*0.999+0.001*maxx)
                self.mm_new.assign(self.mm_new*0.99+0.01*maxx)


            all_vars = model.trainable_variables
            grads = tape.gradient(loss, all_vars)

            optimizer.apply_gradients(zip(grads, all_vars))



            return loss_q, loss, tf.reduce_mean(entr)
        return train


    @tf.function
    def train_p(self, states, policy):

        aug_states = states
        # --- Policy loss (variance reduction + entropy penalty) ---
        with tf.GradientTape(persistent=True) as tape_pi:
            pi = self.policy_net(aug_states, training=True)

            qa = []
            for i in range(self.n_heads):
                qa.append(tf.convert_to_tensor(self.q_heads[i](states, training=True)))

            qq = tf.reduce_mean(tf.stack(qa),axis=0)

            loss_pi = 0.0
            difmm = 0.0
            for i in range(self.n_outs):
                q = qq[i]
                #minx = tf.reduce_min(q, axis=-1)
                #maxx = tf.reduce_max(q, axis=-1)
                #mm = maxx-minx
                #alpha = (mm+1.0)/125
                #qe = tf.exp(tf.clip_by_value((q-maxx[:,None])/alpha[:,None],-6.0,0.0))
                #qsum = tf.reduce_sum(qe,axis=-1)
                #qe = qe / (qsum[:,None])
                #log = tf.math.log(qe)

                #logpi = tf.math.log(y_pi)
                #v1 = tf.reduce_sum(qe*(log - logpi),axis=-1)
                #v2 = tf.reduce_sum(y_pi*(logpi -  log),axis=-1)
                #loss_pi  += tf.reduce_mean(v1+v2)
                y_pi = 1e-12+pi[i]

                baseline = tf.reduce_sum(tf.stop_gradient(y_pi) * q, axis=-1, keepdims=True)
                advantage = q - baseline
                log_pi = tf.math.log(y_pi)
                policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * advantage, axis=-1))
                entropy = -tf.reduce_sum(y_pi * log_pi, axis=-1)
                logp = tf.math.log(policy+1e-12)


                penalty = tf.reduce_mean(tf.nn.relu(-8.0 - log_pi))+tf.nn.relu(tf.reduce_sum(policy*(logp - log_pi),axis=-1)-0.15)*5

                loss_pi += policy_loss + penalty


            loss_pi /=self.n_outs

        grads_pi = tape_pi.gradient(loss_pi, self.policy_net.trainable_variables)
        self.optimizer_pi.apply_gradients(zip(grads_pi, self.policy_net.trainable_variables))

        return loss_pi

    @tf.function
    def _update_target_networks(self):
        tau = self.tau_v
        for tq, q in zip(self.target_q_heads, self.q_heads):
            for t, o in zip(tq.trainable_variables, q.trainable_variables):
                t.assign(tau * t + (1 - tau) * o)
        tau = 0.995
        for t, o in zip(self.target_pi.trainable_variables, self.policy_net.trainable_variables):
                t.assign(tau * t + (1 - tau) * o)



    def learn_all(self):
        if not hasattr(self,"__val__"):
            self.__val__ = 0
        else: self.__val__+=1

        batch = self.buffer.sample_batch(self.T)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        loss_q,  loss, entropy = self.opt[self.__val__%self.n_heads](states,actions,rewards,next_states, dones)
        loss_pi = 0


        batch = self.buffer.sample_batch(self.T)
        loss_pi = self.train_p(batch['states'], batch['policy'])

        if self.__val__%int(self.learn_target)==0:
            self._update_target_networks()

        return loss_q, loss_pi, entropy, loss

    def step(self):
        prev_st = self.env.state()
        act, policy= self.get_action(prev_st)

        state, reward, done = self.env.get_state(act)


        # Compute entropy for buffer
        pi = policy
        entropy = -np.sum(pi * np.log(pi + 1e-8))

        self.buffer.add(state=prev_st, action=act, reward= np.clip(reward, -1,1), next_state=state, done=done, policy=policy)
        self.cur_reward += reward
        loss_q, loss_pi, entropy, loss = 0, 0, 0 ,0

        if self.index > self.T*4  and self.index % self.num_learn == 0:
            loss_q, loss_pi, entropy, loss = self.learn_all()


        if done:
            self.i_out = (self.i_out+1)%self.n_outs
            self.mean_index = self.mean_index*0.95 + self.env.index * 0.05
            self.env.env_reset()
            self.cur_reward10 += self.cur_reward


            self.max_reward = self.max_reward*0.9 + self.cur_reward*0.1
            self.last_reward = self.last_reward*0.95 + self.cur_reward*0.05

            self.cur_reward = 0
            self.num_games += 1



            if self.num_games >= 50:
                print(self.num_games)
                self.cur_reward100 = self.cur_reward10 / 50
                self.xindex.append(self.index)
                self.rewardy.append(self.cur_reward100)
                self.cur_reward10 = 0
                self.num_games = 0
                self.num_learn = int((self.mean_index/400))+1
                self.learn_target = int((self.mean_index/100))+1

                self.optimizer[0].learning_rate = self.optimizer[0].learning_rate*0.995
                for iopt in range(1, self.n_heads):
                    self.optimizer[iopt].learning_rate = self.optimizer[0].learning_rate

                self.optimizer_pi.learning_rate = self.optimizer[0].learning_rate


                if(self.optimizer[0].learning_rate<2e-6):
                    self.optimizer[0].learning_rate = 2e-6


        if self.index % 5000 == 0 and self.index>5000:

            timed = (datetime.now() - self.start_time).total_seconds()
            cur_rew = self.cur_reward10/(50 if self.num_games == 0 else self.num_games)
            sprint = f"Step {self.index}, mr {self.max_reward:.3f} ml {self.last_reward:.3f} tau {self.tau_v.numpy():.5f}, "
            sprint1 = f"Time: {int(timed)}s, Rew_num {cur_rew:.3f}, Reward100: {self.cur_reward100:.3f}, loss_q {loss_q:.4f}, "
            sprint2 = f"entr {entropy:.3f}, optimizer {self.optimizer[0].learning_rate.numpy():.5e}  "
            sprint3 = f"mmold {self.mm_old.numpy():.4f}  mmnew {self.mm_new.numpy():.4f} learn_targ {self.learn_target:.4f} learn_targ {self.num_learn:.4f}  mean_ind {self.mean_index}"
            sprint = sprint+sprint1+sprint2 + sprint3

            self.logger.log(sprint)
            print(sprint)

            if self.index % 50000 == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.xindex, self.rewardy)
                plt.xlabel('Steps')
                plt.ylabel('Avg Reward (100 games)')
                plt.title('Training Progress')
                plt.savefig("./out/figure_rew9.png")
                plt.close()

                df = pd.DataFrame({'x': self.xindex, 'y': self.rewardy})
                df.to_excel('./out/file_name10.xlsx', index=False)

                #self.show()

        self.index += 1

    def show(self):
        cv2.imshow(self.nwin, self.env.get_image())
        if cv2.waitKey(1) == 27:
            self.flag = not self.flag

if __name__ == "__main__":
    logger = Logger()
    env = EnvironmentWrapper()
    agent = SACAgent(env, logger)

    try:
        for i in range(100_000_000):
            agent.step()
    finally:
        cv2.destroyAllWindows()
