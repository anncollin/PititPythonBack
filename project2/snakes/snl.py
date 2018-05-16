# env.observation_space.shape
# self.env.action_space.n
# env.reset()
# env.render()
# env.step(action)
import sys
sys.path.append("../../project1")
from board import Board
from autoplay import _play_turn
from random import randrange
import numpy as np
from dies import get_all_dice
from gui import draw_board
import pygame


class Shape:
    def __init__(self, n=0, shape=(0,)):
        self.shape = shape
        self.n = n

    def sample(self):
        # arr = np.array([0]*self.n)
        # arr[randrange(0, self.n)] = 1
        return randrange(0, self.n)


class SnlEnv:
    def __init__(self, board):
        self.observation_space = Shape(n=15, shape=(15, ))
        self.action_space = Shape(n=3)
        self.board = board
        self.tile = 0
        self.all_dice = np.array(get_all_dice())
        pygame.init()
        self.display = pygame.display.set_mode((2760//2, 480//2))
        self.dice = [-1]*15
        self.type_board = [x.type for x in board.board]

    def reset(self):
        self.tile = 0
        self.dice = [-1] * 15
        return self.get_state()

    def render(self):
        img = draw_board(self.type_board,
                         [d + 1 for d in self.dice],
                         [0]*15,
                         1,
                         "empty",
                         "empty" + ".png",
                         surf=pygame.display.get_surface(),
                         render=True)
        pygame.display.update()

    def step(self, action):
        prev = self.tile
        self.dice[self.tile] = action
        self.tile, reward = _play_turn(self.tile, self.all_dice[action], self.board)
        if self.tile == 14:
            reward = 100
        else:
            reward =(self.tile-prev) - reward

        return self.get_state(), reward, self.tile == 14, None  # or == 14?

    def get_state(self):
        arr = np.zeros(15)
        arr[self.tile] = 1
        return arr # np.array([self.tile, self.board.get_tile(self.tile).type])  # + self.type_board)  # self.board.get_tile(self.tile).type])

    def quit(self):
        pygame.display.quit()
