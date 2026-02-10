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
        #self.env = gym.make("MsPacman-v4", render_mode="rgb_array")

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


class ReplayBuffer:
    def __init__(self, state_shape, n_actions, buffer_size=40000):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.buffer_size = buffer_size

        # Буферы данных
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((buffer_size, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.policy = np.zeros((buffer_size, n_actions), dtype=np.float32)

        # Веса и кумулятивные веса
        self.weights = np.zeros(buffer_size, dtype=np.float32)
        self.cum_weights = np.zeros(buffer_size, dtype=np.float32)  # cum_weights[i] = sum(weights[0..i])

        self.idx = 0
        self.full = False
        self.current_game_start = 0
        self.total_weight = 0.0  # суммарный вес всех шагов

    def add(self, state, action, reward, next_state, done, policy=None):
        if policy is None:
            policy = np.zeros(self.n_actions, dtype=np.float32)

        i = self.idx
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = bool(done)
        self.policy[i] = policy

        # Начало новой игры?
        if i == 0 or self.dones[i - 1]:
            self.current_game_start = i

        # Завершение игры — обновляем веса
        if done:
            game_length = i - self.current_game_start + 1
            if game_length > 0:
                weight = 1.0 / game_length

                # Обновляем веса шагов игры
                start = self.current_game_start
                end = i + 1
                old_weights = self.weights[start:end].copy()
                self.weights[start:end] = weight

                # Инкрементально обновляем total_weight
                old_sum = np.sum(old_weights)
                new_sum = weight * game_length
                self.total_weight += (new_sum - old_sum)

                # Обновляем кумулятивные веса от start до конца буфера
                # (можно оптимизировать через Fenwick tree, но для simplicity — полное обновление)
                max_idx = self.buffer_size if self.full else self.idx + 1
                self.cum_weights[:max_idx] = np.cumsum(self.weights[:max_idx])

        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True

    def sample_batch(self, batch_size):
        max_idx = self.buffer_size if self.full else self.idx
        if max_idx == 0 or self.total_weight == 0:
            return None

        # Генерируем случайные значения в [0, total_weight)
        # Важно: использовать np.nextafter для избежания равенства total_weight
        random_vals = np.random.uniform(0, self.total_weight, size=batch_size)

        # Обрезаем значения, которые могут быть >= total_weight из-за float ошибок
        random_vals = np.clip(random_vals, 0, self.total_weight * (1 - np.finfo(np.float32).eps))

        # Бинарный поиск
        indices = np.searchsorted(self.cum_weights[:max_idx], random_vals, side='right')

        # Защита от выхода за границы
        indices = np.clip(indices, 0, max_idx - 1)

        return {
            'states': tf.cast(self.states[indices], tf.uint8),
            'next_states': tf.cast(self.next_states[indices], tf.uint8),
            'actions': tf.cast(self.actions[indices], tf.int32),
            'rewards': tf.cast(self.rewards[indices], tf.float32),
            'dones': tf.cast(self.dones[indices], tf.float32),
            'policy': tf.cast(self.policy[indices], tf.float32),
            'indices': indices
        }




def get_ind_r(x):
    CONST_MINV = 0.0011 / (1 - 0.0011 * len(x))
    x = x + CONST_MINV
    f = np.cumsum(x) / (1 + CONST_MINV * len(x))
    v = np.random.random()
    index = np.digitize(v, f)
    return min(index, len(x) - 1)




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
            buffer_size=200000


        )

        self.T = 64
        self.gamma = tf.Variable(0.99, trainable=False)
        self.index = 0
        self.cur_reward = 0.0
        self.num_games = 0
        self.cur_reward10 = 0.0
        self.cur_reward100 = -10000.0
        self.max_reward = 1.0
        self.last_reward = 1.0
        self.all_max_rew = -100000
        self.num_learn = 1
        self.num_actor = 1

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
        self.change_max = 0
        self.entropy_sgl = 1.0
        self.loss_q_ma = tf.Variable(0.0, trainable=False)




        self.nwin = "Main"
        self.flag = False
        #cv2.namedWindow(self.nwin)
        #cv2.setMouseCallback(self.nwin, self.capture_event)

        from tensorflow.keras.layers import Dropout, Add, Input, Conv2D, Flatten, Dense, Reshape, UpSampling2D, Concatenate, BatchNormalization, Lambda, LeakyReLU
        from tensorflow.keras import Model
        from tensorflow.keras import layers

        from keras.layers import Lambda
        from tensorflow.keras.utils import register_keras_serializable
        @register_keras_serializable()
        def rescale(x):
            y = tf.cast(x, tf.float32) / 256.0
            return tf.transpose(y, perm=[0, 2, 3, 1])


         # ===== IMPALA ENCODER (REPLACEMENT START) =====
        def impala_residual_block(x, filters):
            residual = x
            x = Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = Conv2D(filters, 3, padding='same')(x)
            x = Add()([x, residual])
            return LeakyReLU(0.02)(x)

        def impala_block(x, filters):
            # Слой уменьшения размерности
            x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
            x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

            # Два остаточных модуля (Residual blocks)
            for _ in range(2):
                residual = x
                x = layers.LeakyReLU()(x)
                x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
                x = layers.LeakyReLU()(x)
                x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
                x = layers.Add()([x, residual])
            return x

        # Encoder
        inp = Input(shape=self.shape_state, dtype='uint8', name='inp')
        x = Lambda(rescale)(inp)

        # Convolutional stack
        #x = Conv2D(16, 8, strides=4, padding='same', activation='relu')(x)  # 96→24
        #x = Conv2D(32, 4, strides=2, padding='same', activation='relu')(x)  # 24→12

        # Residual blocks (x2)
        x = impala_block(x, 32)
        x = impala_block(x, 64)
        x = impala_block(x, 128)

        # Final convolution
        x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)  # 12→6

        flat = Flatten()(x)


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

        self.q_heads[0].summary()

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

        self.games_since_last_eval = 0

        self.eval_cycle_games = 50


        self.hyperparams = {
            'gamma': 0.99,
            'model_freq': 2,
            'target_freq': 1,
            'learning_rate': 0.0002
        }
        self.num_learn = self.hyperparams['model_freq']
        self.learn_target = self.hyperparams['target_freq']
        self.optimizer[0].learning_rate.assign(self.hyperparams['learning_rate'])

        # Logging
        # Logging
        self.xindex = []
        self.rewardy = []
        self.loss_q_history = []   # ← новые списки
        self.loss_pi_history = []
        self.entropy_history = []
        self.loss_xindex = []      # шаги для потерь
        self.loss_q_sgl = 1e-5
        self.loss_pi_sgl = 0.0


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
                q_list = model(aug_states, training=True)  # list of [B, A]
                pi_next = tf.reduce_mean(self.target_pi(aug_next_states, training=True), axis=0)  # [n_outs, B, A]

                qt_list = []
                for i in range(self.n_heads):
                    qt_list.append(tf.convert_to_tensor(self.target_q_heads[i](aug_next_states, training=True)))
                qt_mean = tf.reduce_mean(tf.stack(qt_list), axis=0)  # [n_outs, B, A]

                total_loss = 0.0
                td_errors_list = []
                entr = 0.0

                for i in range(self.n_outs):
                    q = q_list[i]  # [B, A]
                    q_a = tf.gather_nd(q, batch_dims=1, indices=actions)  # [B]

                    log_pi = tf.math.log(pi_next[i] + 1e-9)
                    entropy = -tf.reduce_sum(pi_next[i] * log_pi, axis=-1) / self.log_nact  # [B]

                    v_target = tf.reduce_sum(pi_next[i] * qt_mean[i], axis=-1) + entropy * 0.001  # [B]
                    y = rewards + self.gamma * (1 - dones) * v_target  # [B]

                    td_error = q_a - y  # [B]
                    loss_q = tf.reduce_mean(tf.square(td_error))
                    total_loss += loss_q

                    #maxx = tf.reduce_max(v_target)
                    #minx = tf.reduce_min(v_target)
                    #self.mm_old.assign(self.mm_old * 0.999 + 0.001 * maxx)
                    #self.mm_new.assign(self.mm_new * 0.999 + 0.001 * minx)
                    entr = entr + tf.reduce_mean(entropy)
                entr /= self.n_outs
                avg_loss = total_loss / self.n_outs

            self.loss_q_ma.assign(self.loss_q_ma * 0.999 + avg_loss * 0.001)

            all_vars = model.trainable_variables
            grads = tape.gradient(avg_loss, all_vars)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            loss_ratio = avg_loss / (self.loss_q_ma + 1e-12)
            if loss_ratio < 0.5:  # текущая потеря < 30% от средней → подозрение на переобучение
                # Уменьшаем градиенты пропорционально "подозрительности"
                scale_factor = tf.maximum(0.02, loss_ratio)**1.5  # минимум 0.2 для сохранения обучения
                grads = [g * scale_factor for g in grads]


            optimizer.apply_gradients(zip(grads, all_vars))

            return avg_loss, entr
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
                y_pi = 1e-12+pi[i]

                baseline = tf.reduce_sum(tf.stop_gradient(y_pi) * q, axis=-1, keepdims=True)
                advantage = q - baseline
                log_pi = tf.math.log(y_pi)
                policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * advantage, axis=-1))
                entropy = -tf.reduce_sum(y_pi * log_pi, axis=-1)
                logp = tf.math.log(policy+1e-12)

                pen1 =  tf.reduce_sum(policy*(logp - log_pi),axis=-1)
                pen2 =  tf.reduce_sum(y_pi*(log_pi - logp),axis=-1)
                penalty = tf.reduce_mean(tf.nn.softplus(-7.5 - log_pi),axis=-1) + tf.nn.softplus(pen1 + pen2-0.3)


                loss_pi += policy_loss + tf.reduce_mean(penalty)


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
        if not hasattr(self, "__val__"):
            self.__val__ = 0
        else:
            self.__val__ += 1

        batch = self.buffer.sample_batch(self.T)
        if batch is None:
            return 0.0, 0.0

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        loss_q, _ = self.opt[self.__val__ % self.n_heads](
            states, actions, rewards, next_states, dones
        )

        batch_pi = self.buffer.sample_batch(self.T)
        loss_pi = 0.0
        if batch_pi is not None:
            loss_pi = self.train_p(batch_pi['states'], batch_pi['policy'])

        if self.__val__ % int(self.learn_target) == 0:
            self._update_target_networks()

        return loss_q, loss_pi


    def step(self):
        prev_st = self.env.state()
        act, policy= self.get_action(prev_st)

        state, reward, done = self.env.get_state(act)


        # Compute entropy for buffer
        pi = policy
        entropy = -np.sum(pi * np.log(pi + 1e-8)) / self.log_nact
        self.entropy_sgl = self.entropy_sgl*0.995 + entropy*0.005
        self.cur_reward += reward

        episode_reward = self.cur_reward if done else None
        self.buffer.add(state=prev_st, action=act, reward= np.clip(reward, -1,1), next_state=state, done=done,
                        policy=policy)

        # В начале step()
        loss_q, loss_pi = 0.0, 0.0

        # При обучении:
        if self.index > self.T*4 and self.index % int(self.num_learn) == 0:
            loss_q, loss_pi = self.learn_all()
            self.loss_q_sgl = self.loss_q_sgl*0.995 + loss_q *0.005
            self.loss_pi_sgl = self.loss_pi_sgl*0.995 + loss_pi *0.005



            if self.index%(self.num_learn*1000)==0:
                self.loss_q_history.append(self.loss_q_sgl)
                self.loss_pi_history.append(self.loss_pi_sgl)
                self.loss_xindex.append(self.index)
                self.entropy_history.append(self.entropy_sgl)


        if done:
            self.i_out = (self.i_out+1)%self.n_outs
            self.mean_index = self.mean_index*0.95 + self.env.index * 0.05
            self.env.env_reset()
            self.cur_reward10 += self.cur_reward


            self.max_reward = self.max_reward*0.9 + self.cur_reward*0.1
            self.last_reward = self.last_reward*0.95 + self.cur_reward*0.05
            if self.all_max_rew < self.last_reward:
                self.all_max_rew = self.last_reward
                self.change_max+=1

            self.cur_reward = 0


            self.num_games += 1
            self.games_since_last_eval += 1

            if self.games_since_last_eval >= self.eval_cycle_games:

                self.learn_target = max(int((self.mean_index / 400)), 2)
                self.num_learn =   max(int((self.mean_index  / 100)), 2)
                self.num_learn = min(self.T,self.num_learn)

                #self.num_actor  = int((self.mean_index / 200)) + 1

                current_avg_reward = self.cur_reward10 / self.eval_cycle_games

                # Обновляем статистику
                self.cur_reward100 = current_avg_reward
                self.xindex.append(self.index)
                self.rewardy.append(self.cur_reward100)

                # Сброс счётчиков
                self.cur_reward10 = 0.0
                self.games_since_last_eval = 0

                print(f"Reward100: {self.cur_reward100:.3f}")


                # Обновляем LR
                self.optimizer[0].learning_rate.assign(self.optimizer[0].learning_rate * 0.94)
                for i in range(1, self.n_heads):
                    self.optimizer[i].learning_rate.assign(self.optimizer[0].learning_rate)
                self.optimizer_pi.learning_rate.assign(self.optimizer[0].learning_rate)

                if self.optimizer[0].learning_rate.numpy() < 5e-6:
                    self.optimizer[0].learning_rate.assign(5e-6)



        if self.index % 5000 == 0 and self.index>5000:

            timed = (datetime.now() - self.start_time).total_seconds()
            cur_rew = self.cur_reward10/(50 if self.num_games == 0 else self.num_games)
            sprint = f"Step {self.index}, mr {self.max_reward:.3f} ml {self.last_reward:.3f} tau {self.tau_v.numpy():.5f}, "
            sprint1 = f"Time: {int(timed)}s, Rew_num {cur_rew:.3f}, Reward100: {self.cur_reward100:.3f}, loss_q {loss_q:.4f}, "
            sprint2 = f"entr {entropy:.3f}, optimizer {self.optimizer[0].learning_rate.numpy():.5e}  "
            sprint3 = f"loss_q_ma {self.loss_q_ma.numpy():.5e} learn_targ {self.learn_target:.4f} num_learn {self.num_learn:.4f}  mean_ind {self.mean_index}"
            sprint = sprint+sprint1+sprint2 + sprint3

            self.logger.log(sprint)
            print(sprint)

        if self.index % 50000 == 0 and self.index>5000:
           # Сохранение графика наград
            plt.figure(figsize=(10, 5))
            plt.plot(self.xindex, self.rewardy)
            plt.xlabel('Steps')
            plt.ylabel('Avg Reward (50 games)')
            plt.title('Training Progress')
            plt.savefig("./out/reward_progress.png")
            plt.close()

            # Сохранение графиков потерь
            if self.loss_xindex:
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.plot(self.loss_xindex, np.log(np.array(self.loss_q_history)))
                plt.xlabel('Steps')
                plt.ylabel('Critic Loss (Q)')
                plt.title('Critic Loss')

                plt.subplot(1, 3, 2)
                plt.plot(self.loss_xindex, self.loss_pi_history)
                plt.xlabel('Steps')
                plt.ylabel('Actor Loss (π)')
                plt.title('Actor Loss')

                plt.subplot(1, 3, 3)
                plt.plot(self.loss_xindex, self.entropy_history)
                plt.xlabel('Steps')
                plt.ylabel('Entropy')
                plt.title('Entropy')



                plt.tight_layout()
                plt.savefig("./out/losses1.png")
                plt.close()

            # Сохранение данных в Excel
            df_reward = pd.DataFrame({'step': self.xindex, 'reward': self.rewardy})
            df_reward.to_excel('./out/rewards1.xlsx', index=False)

            df_loss = pd.DataFrame({
                'step': self.loss_xindex,
                'loss_q': self.loss_q_history,
                'loss_pi': self.loss_pi_history,
                'entropy': self.entropy_history
            })
            df_loss.to_excel('./out/losses1.xlsx', index=False)

        self.index += 1

    def show(self):
        cv2.imshow(self.nwin, self.env.get_image())
        if cv2.waitKey(1) == 27:
            self.flag = not self.flag

if __name__ == "__main__":

    os.makedirs('./checkpoints', exist_ok=True)
    logger = Logger()
    env = EnvironmentWrapper()
    agent = SACAgent(env, logger)

    try:
        for i in range(100_000_000):
            agent.step()
    finally:
        cv2.destroyAllWindows()
