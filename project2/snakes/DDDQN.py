import gym
import numpy as np
import random
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Input, multiply, add, subtract, Lambda
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from collections import deque
from time import sleep
from snl import SnlEnv
from board import Board


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
        self.memory = deque(maxlen=100000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.learning_rate = 0.005
        self.tau = .5

        self.action_space = (1, self.env.action_space.n)
        self.model = self.create_model(self.env)
        self.target_model = self.create_model(self.env)

    @staticmethod
    def create_model(env):
        optimizer = Adam(lr=0.001) #RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)  #
        state_shape = env.observation_space.shape

        # Layers for the model
        state_input = Input(state_shape, name='frames')
        actions_input = Input((env.action_space.n, ), name='mask')
        hidden1 = Dense(16, activation="relu", name='hidden1')(state_input)
        hidden2 = Dense(32, activation="relu", name='hidden2')(hidden1)

        # Right part
        hidden3 = Dense(16, activation="relu", name='hidden3')(hidden2)
        output = Dense(env.action_space.n, name='output')(hidden3)

        def layer_mean(xin):
            ss = K.mean(xin, axis=1, keepdims=True)  # compute mean
            ss = K.repeat_elements(ss, env.action_space.n, axis=1)  # repeat mean
            return ss
        mean = Lambda(lambda xin: layer_mean(xin), name="mean")(output)
        sub_output = subtract([output, mean], name="sub_output")

        # Left Part
        para_hidden3 = Dense(8, activation="relu", name='para_hidden3')(hidden2)
        para_output = Dense(1, name='para_output')(para_hidden3)

        def repeat(xin):
            ss = K.repeat_elements(xin, env.action_space.n, axis=1)  # repeat mean
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
            return self.env.action_space.sample()

        return np.argmax(self.model.predict([state,
                                             np.ones(self.action_space)])[0])

    def remember(self, state, action, reward, tot_reward, new_state, done):
        self.memory.append([state, action, reward, tot_reward, new_state, done])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        states = np.empty((batch_size,) + self.env.observation_space.shape, dtype="float32")
        targets = np.empty((batch_size, 3))  # +self.action_space())
        actions = np.empty((batch_size, 3))
        samples = random.sample(self.memory, batch_size)
        for i in range(len(samples)):
            state, action, reward, tot_reward, new_state, done = samples[i]
            # np.append(states, state)
            target = np.zeros(self.env.action_space.n)
            if done:
                target[action] = reward
            else:
                q_argmax = np.argmax(self.model.predict([new_state,
                                                         np.ones(self.action_space)])[0])
                q_vec = np.zeros(self.action_space)
                q_vec[0][q_argmax] = 1
                q_future = self.target_model.predict([new_state,
                                                      q_vec])[0][q_argmax]
                target[action] = reward + q_future * self.gamma
            # np.append(targets, target[0])
            in_actions = np.zeros(self.action_space)
            in_actions[0][action] = 1
            states[i] = state[0]
            targets[i] = target[0]
            actions[i] = in_actions[0]
            # self.model.fit([state, in_actions], np.array([target]), epochs=1, verbose=0)
        self.model.fit([states, actions], targets, epochs=1, verbose=0, batch_size=batch_size)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, fn):
        self.model.save(fn)
        self.model.save_weights(fn + ".weights")


def replay():
    # input("Continue ?")
    while True:
        # env = SnlEnv(
        #     Board({"trap1": [9, 13], "trap2": [8, 12, 4], "trap3": [1, 10, 11]}, circling=True)
        # )
        env = SnlEnv(
            Board({"trap1": [], "trap2": [], "trap3": []}, circling=False)
        )

        # for i in range(100):
        # print(str(i)+"============================")
        model = DQN.create_model(env)
        # model.load_weights(str(i)+"success.model")
        model.load_weights("last_model.weights")
        # Replaying to watch what it looks like
        env.reset()
        for xi in range(15):

            # print(cur_state)
            env.tile = xi
            cur_state = env.get_state().reshape(1, env.observation_space.n)

            # print(cur_state)
            action = np.argmax(model.predict([cur_state, np.ones((1, env.action_space.n))])[0])
            env.step(action)

        env.render()
        sleep(1)
        got = input("Stop ? [y/N] ")
        env.quit()
        if "y" in got or "Y" in got:
            break


def main():
    # env = SnlEnv(
    #     Board({"trap1": [9, 13], "trap2": [8, 12, 4], "trap3": [1, 10, 11]}, circling=True)
    # )
    env = SnlEnv(
        Board({"trap1": [], "trap2": [], "trap3": []}, circling=False)
    )

    trials = 100000
    trial_len = 800
    iter_limit = -1000
    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    print("init done")
    total_ok = 0
    steps = 0
    while True:
        cur_state = env.reset().reshape(1, env.observation_space.n)
        total_reward = 0
        # print(iter_limit)
        for step in range(trial_len):
            steps += 1
            # env.render()

            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            total_reward += reward
            new_state = new_state.reshape(1, env.observation_space.n)
            dqn_agent.remember(cur_state, action, reward, total_reward, new_state, done)
            dqn_agent.replay(32)  # internally iterates default (prediction) model
            if steps % 150:
                dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break

        if total_reward > iter_limit:
            iter_limit = total_reward

        # dqn_agent.replay()

        dqn_agent.save_model("last_model")
        print("reward ", total_reward)

    print(total_ok)
    env.quit()


if __name__ == "__main__":
    random.seed(456)
    # main()
    replay()