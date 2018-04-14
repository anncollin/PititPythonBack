import gym
import numpy as np
import random
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Input, multiply
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from collections import deque
from time import sleep


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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .5

        self.action_space = (1, self.env.action_space.n)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)  # Adam(lr=self.learning_rate)
        state_shape = self.env.observation_space.shape
        state_input = Input(state_shape, name='frames')
        actions_input = Input((self.env.action_space.n, ), name='mask')
        hidden1 = Dense(16, activation="relu", name='hidden1')(state_input)
        hidden2 = Dense(32, activation="relu", name='hidden2')(hidden1)
        hidden3 = Dense(64, activation="relu", name='hidden3')(hidden2)
        output = Dense(self.env.action_space.n, name='output')(hidden3)
        filtered_output = multiply([output, actions_input], name='filtered_output')
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
        # batch_size = 8
        if len(self.memory) < batch_size:
            return

        states = np.empty((1, 8), dtype="float32")
        targets = np.empty((1, 4))
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, tot_reward, new_state, done = sample
            np.append(states, state)
            target = np.zeros(self.env.action_space.n)
            if done or tot_reward >= 200:
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
            self.model.fit([state, in_actions], np.array([target]), epochs=1, verbose=0)
        # self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, fn):
        self.model.save(fn)
        self.model.save_weights(fn + ".weights")



def replay():
    input("Continue ?")
    env = gym.make("LunarLander-v2")

    for i in range(21):
        print(str(i)+"============================")
        model = load_model(str(i)+"success.model", custom_objects={'huber_loss': huber_loss})
        # model = load_model("last_model", custom_objects={'huber_loss': huber_loss})
        # model.load_weights("success.model.weights")
        for _ in range(5):
            # Replaying to watch what it looks like
            cur_state = env.reset()
            # print(cur_state)
            cur_state = cur_state.reshape(1, 8)
            score = 0
            total_reward = 0
            for step in range(1000):
                # print(cur_state)
                action = np.argmax(model.predict([cur_state, np.ones((1, 4))])[0])
                new_state, reward, done, _ = env.step(action)
                total_reward += reward
                new_state = new_state.reshape(1, 8)
                score += reward
                env.render()
                sleep(0.001)
                cur_state = new_state
                if done or total_reward >= 200:
                    break
            print(total_reward)


def main():
    env = gym.make("LunarLander-v2")
    gamma = 0.9
    epsilon = .95

    trials = 10000
    trial_len = 800
    iter_limit = -1000
    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    total_ok = 0
    steps = 0
    for trial in range(trials):
        cur_state = env.reset().reshape(1, 8)
        total_reward = 0
        learned = 0
        print(iter_limit)
        for step in range(trial_len):
            steps += 1
            env.render()

            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            total_reward += reward
            new_state = new_state.reshape(1, 8)
            dqn_agent.remember(cur_state, action, reward, total_reward, new_state, done)
            dqn_agent.replay(32)  # internally iterates default (prediction) model
            if steps % 2000:
                dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done or total_reward >= 200:
                break

        if total_reward > iter_limit:
            iter_limit = total_reward

        # dqn_agent.replay()
        dqn_agent.save_model("last_model")

        print(total_reward)
        if total_reward > 0:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model(str(total_ok)+"success.model")
            total_ok += 1
            if total_ok > 100:
                break
        else:
            print("Failed to complete in trial {}".format(trial))
        dqn_agent.target_train()
        if steps > 100000:
            print(total_ok)
            break
    print(total_ok)


if __name__ == "__main__":
    random.seed(456)
    main()
    replay()
