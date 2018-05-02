from board import make_board
from MDP import markov_decision
import sys

'''
    @args: a list containing which traps and 
    
    @return: a tuple containing in 
        position 0: an array with the number of expected throws 
                        to reach the end from each square
        position 1: an array with which die is the best one to use 
                        for each square
'''
def markovDecision(args):
    if len(args) > 15:
        print("Unsupported input length")
        return [], []
    board = make_board(args)
    return markov_decision(board)


if __name__ == "__main__":
    print([int(x) for x in sys.argv[1:]])
    Expec, Dice = markovDecision([int(x) for x in sys.argv[1:]])
    print("Square nmb : "+''.join(["{:5.0f} ".format(x) for x in list(range(1, 16))]))
    print("Expectation: "+''.join(["{:5.2f} ".format(x) for x in Expec]))
    print("Dice       : "+''.join(["{:5.0f} ".format(x) for x in Dice]))
