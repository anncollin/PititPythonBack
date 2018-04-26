import gym
import numpy as np
import random
from keras.models import load_model, Model
from keras.layers import Dense, Input, multiply, add, subtract, Lambda, Conv2D, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import np_utils
from time import sleep
from my_utils.RingBuffer import RingBuffer
from my_utils.action_space import ToDiscrete
import multiprocessing

# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = RingBuffer(10000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .5

        self.model = self.create_model()
        self.target_model = self.create_model()

    @staticmethod
    def n_actions():
        return 14

    @staticmethod
    def action_space():
        return 1, DQN.n_actions()

    @staticmethod
    def to_cat(x):
        assert x < DQN.n_actions()
        v = np.zeros(6, dtype="int")
        v[x % 4] = 1
        if x < 4: v[4] = 1
        else: v[5] = 1
        return v

    @staticmethod
    def preprocess(img):
        def to_grayscale(img):
            return np.mean(img, axis=2).astype(np.uint8)

        def downsample(img):
            return img[::4, ::4]

        return to_grayscale(downsample(img)).reshape(1, 1, 56, 64)

    @staticmethod
    def create_model():
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)  # Adam(lr=self.learning_rate)
        state_shape = (1, 56, 64)
        n_actions = DQN.n_actions()

        # Layers for the model
        state_input = Input(state_shape, name='frames')
        actions_input = Input((n_actions, ), name='mask')
        normalized = Lambda(lambda x: x / 255.0)(state_input)
        conv1 = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation="relu", name='conv1')(normalized)
        conv2 = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation="relu", name='conv2')(conv1)
        flat = Flatten(name="flat")(conv2)

        # Right part
        hidden = Dense(256, activation="relu", name='hidden')(flat)
        output = Dense(n_actions, name='output')(hidden)

        def layer_mean(xin):
            ss = K.mean(xin, axis=1, keepdims=True)  # compute mean
            ss = K.repeat_elements(ss, n_actions, axis=1)  # repeat mean
            return ss
        mean = Lambda(lambda xin: layer_mean(xin), name="mean")(output)
        sub_output = subtract([output, mean], name="sub_output")

        # Left Part
        para_hidden = Dense(256, activation="relu", name='para_hidden')(flat)
        para_output = Dense(1, name='para_output')(para_hidden)

        def repeat(xin):
            ss = K.repeat_elements(xin, n_actions, axis=1)  # repeat mean
            return ss
        para_repeat_output = Lambda(lambda xin: repeat(xin), name="repeat")(para_output)

        # Sum
        sum_outputs = add([sub_output, para_repeat_output], name="sum_outputs")

        filtered_output = multiply([sum_outputs, actions_input], name='filtered_output')
        model = Model(inputs=[state_input, actions_input], outputs=filtered_output)
        model.compile(loss=huber_loss,
                      optimizer=optimizer)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            # self.env.action_space.sample()
            return random.randrange(self.n_actions())
        return np.argmax(self.model.predict([self.preprocess(state),
                                             np.ones(self.action_space())])[0])

    def remember(self, state, action, reward, tot_reward, new_state, done):
        self.memory.append([self.preprocess(state), action, reward,
                            tot_reward, self.preprocess(new_state), done])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        # states = np.empty((1, 8), dtype="float32")
        # targets = np.empty((1, 4))
        # samples = random.sample(self.memory, batch_size)
        for _ in range(batch_size):
            sample = self.memory.get_random()
            state, action, reward, tot_reward, new_state, done = sample
            target = np.zeros(self.n_actions())
            if done or tot_reward >= 200:
                target[action] = reward
            else:
                q_argmax = np.argmax(self.model.predict([new_state,
                                                         np.ones(self.action_space())])[0])
                q_vec = np.zeros(self.action_space())
                q_vec[0][q_argmax] = 1
                q_future = self.target_model.predict([new_state,
                                                      q_vec])[0][q_argmax]
                target[action] = reward + q_future * self.gamma
            in_actions = np.zeros(self.action_space())
            in_actions[0][action] = 1
            self.model.fit([state, in_actions], np.array([target]), epochs=1, verbose=0)
        # self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, fn):
        self.model.save("aw/"+fn)
        self.model.save_weights("aw/"+fn + ".weights")


# def replay():
#     input("Continue ?")
#     env = gym.make("LunarLander-v2")
#
#     for i in range(21):
#         print(str(i)+"============================")
#         model = load_model(str(i)+"success.model", custom_objects={'huber_loss': huber_loss})
#         for _ in range(5):
#             # Replaying to watch what it looks like
#             cur_state = env.reset()
#             # print(cur_state)
#             cur_state = cur_state.reshape(1, 8)
#             score = 0
#             total_reward = 0
#             for step in range(1000):
#                 # print(cur_state)
#                 action = np.argmax(model.predict([cur_state, np.ones((1, 4))])[0])
#                 new_state, reward, done, _ = env.step(action)
#                 total_reward += reward
#                 new_state = DQN.preprocess(new_state)
#                 score += reward
#                 env.render()
#                 sleep(0.001)
#                 cur_state = new_state
#                 if done or total_reward >= 200:
#                     break
#             print(total_reward)


def main():
    env = gym.make("SuperMarioBros-1-1-v0")
    # env.configure(lock=multiprocessing.Lock())
    wrapper = ToDiscrete()
    env = wrapper(env)

    gamma = 0.9
    epsilon = .95

    trials = 10000
    trial_len = 100000
    iter_limit = -1000
    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    total_ok = 0
    steps = 0
    for trial in range(trials):
        print("start loop")
        cur_state = env.reset()
        total_reward = 0
        learned = 0
        print(iter_limit)
        for step in range(trial_len):
            steps += 1
            # env.render()
            # a = env.action_space.sample()
            # env.step(a)
            # continue

            action = dqn_agent.act(cur_state)
            new_state, reward, done, info = env.step(action)
            print(info)
            if done and 'ignore' in info:
                print("broke free")
                break

            total_reward += reward
            new_state = new_state
            dqn_agent.remember(cur_state, action, reward, total_reward, new_state, done)
            dqn_agent.replay(32)  # internally iterates default (prediction) model
            if steps % 1000:
                dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break

        if total_reward > iter_limit:
            iter_limit = total_reward

        dqn_agent.save_model("last_model")
        env.close()
        # print(total_reward)
        # if total_reward > 0:
        #     print("Completed in {} trials".format(trial))
        #     dqn_agent.save_model(str(total_ok)+"success.model")
        #     total_ok += 1
        #     if total_ok > 100:
        #         break
        # else:
        #     print("Failed to complete in trial {}".format(trial))
        # dqn_agent.target_train()
        if steps > 100000:
            print(total_ok)
            break
    print(total_ok)


if __name__ == "__main__":
    random.seed(456)
    main()
    # replay()
