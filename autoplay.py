from random import randrange, random, seed
from board import Board
from dies import get_all_dice
from MDP import markovDecision


def play(dice, board, n_games=100):
    moves = {}
    seed(123)
    for my_die in dice:
        die_list = [0]*board.goal
        for start in range(board.goal):
            for _ in range(n_games):
                die_list[start] += _play_die(start, my_die, board)
            die_list[start] /= n_games
        moves[my_die] = die_list
    return moves


def _play_die(start, my_die, board):
    actual = start
    n_moves = 0
    while actual != board.goal:
        actual, step_n_moves = _play_turn(actual, my_die, board)
        n_moves += step_n_moves
    return n_moves


def _play_turn(start, my_die, board):
    actual = start
    n_moves = 0
    step = randrange(my_die.adv + 1)
    first = True
    # Advance until end of throw
    while step > 0:
        succ = board.get_successors(actual, first=first)
        actual = succ[randrange(len(succ))]
        first = False
        step -= 1
    n_moves += 1

    # Check for traps
    if random() < my_die.trap_prob:
        trap = board.get_tile(actual)
        n_moves += trap.get_lost_turns()
        actual = trap.get_new_state(actual)
    return actual, n_moves


def play_strategy(dice, board, strategy, n_games):
    moves = [0]*board.goal
    seed(123)
    for start in range(board.goal):
        for _ in range(n_games):
            moves[start] += _play_die_strategy(start, dice, board, strategy)
        moves[start] /= n_games
    return moves


def _play_die_strategy(start, dice, board, strategy):
    actual = start
    n_moves = 0
    while actual != board.goal:
        actual, step_n_moves = _play_turn(actual, dice[strategy(actual)], board)
        n_moves += step_n_moves
    return n_moves


if __name__ == "__main__":
    traps = {'trap1': [], 'trap2': []}
    my_board = Board(traps, circling=True)
    matrix = [0]*3
    iter = 0
    for my_die, x in play(get_all_dice(), my_board, n_games=1000).items():
        matrix[iter] = x
        iter += 1
        print(str(my_die), " = ", str(x))

    strat = [-1]*my_board.goal
    for i in range(my_board.goal):
        min_val = 1000000000
        min_die = -1
        for j in range(3):
            val = matrix[j][i]
            if val < min_val:
                min_val = val
                min_die = j
        strat[i] = min_die
    print(str(strat))
    print(play_strategy(get_all_dice(), my_board, lambda _x: strat[_x], n_games=1000))
    Expec, Dice = markovDecision(my_board)
    mdp_dice = [d-1 for d in Dice]
    print(mdp_dice)
    # print(Expec)
    print(play_strategy(get_all_dice(), my_board, lambda _x: mdp_dice[_x], n_games=1000))
    print("random for fun")
    print(play_strategy(get_all_dice(), my_board, lambda _x: randrange(3), n_games=1000))

