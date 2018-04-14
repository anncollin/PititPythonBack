import gym
import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


from collections import deque
from time import sleep


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.recent = deque(maxlen=500)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.perfect_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
        self.recent.append([state, action, reward, new_state, done])

    def clear_recent(self):
        self.recent.clear()

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        states = np.empty((1, 2), dtype="float32")
        targets = np.empty((1, 3))
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            np.append(states, state)
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            np.append(targets, target[0])

            self.model.fit(state, target, epochs=1, verbose=0, batch_size=batch_size)
        # self.model.fit(states, targets, epochs=1, verbose=0, batch_size=1)

    def replay_perfect(self):
        batch_size = 200
        if len(self.memory) < batch_size:
            batch_size = 32

        states = np.empty((1, 2), dtype="float32")
        targets = np.empty((1, 3))
        for sample in self.recent:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            np.append(states, state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            np.append(targets, target[0])
            self.perfect_model.fit(state, target, epochs=1, verbose=0, batch_size=batch_size)
        # self.perfect_model.fit(states, targets, epochs=10, verbose=0, batch_size=8)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
        self.model.save_weights(fn+".weights")
        self.perfect_model.save("perfect_"+fn)
        self.perfect_model.save_weights("perfect_"+fn + ".weights")


def replay():
    input("Continue ?")
    env = gym.make("MountainCar-v0")

    model = load_model("perfect_success.model")
    model.load_weights("perfect_success.model.weights")
    # model = load_model("target_success.model")

    # model = Sequential()
    # state_shape = env.observation_space.shape
    # model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
    # model.add(Dense(48, activation="relu"))
    # model.add(Dense(24, activation="relu"))
    # model.add(Dense(env.action_space.n))
    # model.compile(loss="mean_squared_error",
    #               optimizer=Adam(lr=0.005))
    # model.load("success.model")

    # Replaying to watch what it looks like
    cur_state = env.reset().reshape(1, 2)
    score = 0
    for step in range(600):
        # print(cur_state)
        action = np.argmax(model.predict(cur_state)[0])
        new_state, reward, done, _ = env.step(action)
        new_state = new_state.reshape(1, 2)
        score += reward
        env.render()
        sleep(0.02)
        cur_state = new_state
        # if done:
        #     break


def main():
    env = gym.make("MountainCar-v0")
    gamma = 0.9
    epsilon = .95

    trials = 10000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    total_ok = 0
    for trial in range(trials):
        cur_state = env.reset().reshape(1, 2)

        for step in range(trial_len):
            env.render()
            new_state = cur_state
            while np.all(cur_state == new_state):
                action = dqn_agent.act(cur_state)
                new_state, reward, done, _ = env.step(action)

            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            if step%10==0:
                dqn_agent.replay()  # internally iterates default (prediction) model
                dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                dqn_agent.replay_perfect()
                break
        dqn_agent.clear_recent()
        # dqn_agent.replay()  # internally iterates default (prediction) model
        # dqn_agent.target_train()  # iterates target model

        if step >= 200 - 1:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            total_ok += 1
            if total_ok > 0:
                break


if __name__ == "__main__":
    main()
    replay()
