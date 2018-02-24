from board import *
from MDP import *

# Initialize a board 
traps    = {'trap1': [1,5,9], 'trap2': [14]}
my_board = Board(traps)
#print(my_board.board)

# Find an optimal strategy
Expec, Dice = markovDecision(my_board)
list = list(range(0,14))
print(Expec, '\n', list, '\n',Dice)
