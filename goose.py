from board import Board
from MDP import *

# todo: modify to satisfy constraints of assignment
# Initialize a board
traps = {'trap1': [3,4,5,6,7], 'trap2': []}
my_board = Board(traps, circling=True)

# Find an optimal strategy
Expec, Dice = markovDecision(my_board)
list = list(range(14))
print(Expec, '\n', list, '\n', Dice)
