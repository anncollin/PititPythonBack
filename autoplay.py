from random import randrange, random, seed
from board import Board
from MDP import markov_decision
import numpy as np
import pprint


# plays a full turn with @my_die on @board starting at @start
# @return the number of moves and the new position
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
        actual = trap.get_new_state(actual, board)
    return actual, n_moves


# plays a full game with @strategy on @board starting at @start
def _play_die_strategy(start, dice, board, strategy):
    actual = start
    n_moves = 0
    while actual != board.goal:
        actual, step_n_moves = _play_turn(actual, dice[strategy(actual)], board)
        n_moves += step_n_moves
    return n_moves


# plays @n_games full games on @board with @strategy
# @return a list with the expectation starting from each square
def play_strategy(board, strategy, n_games=100):
    dice = board.dice
    moves = [0]*board.goal
    for start in range(board.goal):
        for _ in range(n_games):
            moves[start] += _play_die_strategy(start, dice, board, strategy)
        moves[start] /= n_games
    return moves


# finds @very_intelligent (or not) strategy for @board based on @n_games
# @return a list with what die to use for each square
def find_strategy(board, n_games=100, very_intelligent=False):
    dice = board.dice
    use_these = [0]*board.goal
    if not very_intelligent:
        for start in range(board.goal):
            avg_dice = {}
            for my_die in dice:
                strategy = lambda _x: my_die.type-1 if _x == start \
                                 else randrange(len(dice))
                moves = 0
                for _ in range(n_games):
                    moves += _play_die_strategy(start, dice, board, strategy)
                avg_dice[my_die] = moves / n_games
            use_these[start] = min(avg_dice, key=avg_dice.get).type - 1
        return use_these

    else:
        change = True
        while change:  # until no change makes sense anymore
            change = False
            for start in range(board.goal):
                avg_dice = {}
                prev_die = use_these[start]
                for my_die in dice:
                    strategy = lambda _x: my_die.type-1 if _x == start \
                                     else use_these[_x]
                    moves = 0
                    for _ in range(n_games):
                        moves += _play_die_strategy(start, dice, board, strategy)
                    avg_dice[my_die] = moves / n_games
                use_these[start] = min(avg_dice, key=avg_dice.get).type - 1
                if use_these[start] != prev_die:
                    change = True
    return use_these


# Launches @n_games for all opposing @strategies on @board
# @return a dictionary with the results
def battle(board, strategies, n_games=100):
    dice = board.dice
    results = {'wins': [0]*len(strategies), 'equal': 0}
    for _ in range(n_games):
        tot_moves = [0] * len(strategies)
        for ind in range(len(strategies)):
            tot_moves[ind] = _play_die_strategy(0, dice, board, strategies[ind])

        min_move = min(tot_moves)
        winners = sum(np.array(tot_moves) == min_move)

        for move, ind in zip(tot_moves, range(len(tot_moves))):
            if move == min_move:
                results['wins'][ind] += 1./winners
        if winners > 1:
            results['equal'] += 1
    return results


if __name__ == "__main__":
    seed(789)
    traps = {'trap1': [1, 6, 8, 13], 'trap2': [5], 'trap3': []}
    b = Board({"trap1": [9, 13], "trap2": [8, 12, 4], "trap3": [1, 10, 11]}, circling=True)#Board(defaultdict(list), circling=True)#Board({"trap1": [9, 13], "trap2": [8, 12, 4], "trap3": [1, 10, 11]}, circling=True)#Board(traps, circling=True)

    dice1 = find_strategy(b, n_games=1000, very_intelligent=True)
    dice2 = find_strategy(b, n_games=1000, very_intelligent=False)
    _, dice3 = markov_decision(b)
    print("epic battles ensue: ")
    results = battle(b, [
        lambda _x: dice1[_x],
        lambda _x: dice2[_x],
        lambda _x: randrange(3)#dice3[_x]-1
    ], n_games=10000)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)




