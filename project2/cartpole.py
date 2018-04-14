"""
__name__   = predict.py
__author__ = Yash Patel
__description__ = Full prediction code of OpenAI Cartpole environment using Keras
"""

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from random import seed, randrange
from time import sleep


def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(4,), activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    return model


def gather_initial_data(env, min_score=50):
    num_trials = 1000
    sim_steps = 500
    x_train, y_train = [], []

    scores = []
    for _ in range(num_trials):
        observation = env.reset()
        score = 0
        x_sample, y_sample = [], []
        for step in range(sim_steps):
            # action corresponds to the previous observation so record before step
            action = np.random.randint(0, 2)
            one_hot_action = np.zeros(2)
            one_hot_action[action] = 1
            x_sample.append(observation)
            y_sample.append(one_hot_action)

            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        if score > min_score:
            scores.append(score)
            x_train += x_sample
            y_train += y_sample

    x_train, y_train = np.array(x_train), np.array(y_train)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return x_train, y_train


def gather_more_data(env, model, min_score):
    num_trials = 20
    sim_steps = 1000
    x_train, y_train = [], []

    scores = []
    for _ in range(num_trials):
        observation = env.reset()
        score = 0
        x_sample, y_sample = [], []
        for step in range(sim_steps):
            # action corresponds to the previous observation so record before step
            action = np.argmax(model.predict(observation.reshape(1, 4)))
            if randrange(5) > 3:
                action = 1 if action == 0 else 1
            one_hot_action = [0]*2
            one_hot_action[action] = 1
            x_sample.append(observation)
            y_sample.append(one_hot_action)

            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        if score > min_score:
            scores.append(score)
            x_train += x_sample
            y_train += y_sample

    x_train, y_train = np.array(x_train), np.array(y_train)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return x_train, y_train


def predict():
    env = gym.make("CartPole-v0")
    model = create_model()

    for i in range(5):
        print("Iteration {}".format(i))
        x_train, y_train = gather_initial_data(env) if i == 0 else gather_more_data(env, model, prev_min)
        print("Done testing!!!!!!!!!!!!!!====================")
        if len(x_train) == 0:
            continue
        model.fit(x_train, y_train, epochs=5)
        scores = []
        num_trials = 5
        sim_steps = 1000
        for _ in range(num_trials):
            observation = env.reset()
            score = 0
            for step in range(sim_steps):
                action = np.argmax(model.predict(observation.reshape(1, 4)))
                observation, reward, done, _ = env.step(action)
                env.render()
                sleep(0.05)
                score += reward
                if done:
                    break
            scores.append(score)
        prev_min = np.mean(scores)
        print("This is the TESTED mean: {}".format(prev_min))
        prev_min *= 0.5


if __name__ == "__main__":
    seed(456)
    predict()
