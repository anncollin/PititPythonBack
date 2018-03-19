from random import randrange, random, seed
from board import Board
from MDP import markovDecision
import pprint


def play(board, n_games=100):
    dice = board.dice
    moves = {}
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
        actual = trap.get_new_state(actual, board)
    return actual, n_moves


def play_strategy(board, strategy, n_games=100):
    dice = board.dice
    moves = [0]*board.goal
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


def find_strategy(board, n_games=100, very_intelligent=False):
    dice = board.dice
    use_these = [0]*board.goal
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
    if not very_intelligent:
        return use_these


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


def battle(board, strategies, n_games=100):
    dice = board.dice
    results = {'wins': [0]*len(strategies), 'equal': 0}
    for _ in range(n_games):
        actual = [0] * len(strategies)
        while board.goal not in actual: # false if one is at end!
            for ind in range(len(strategies)):
                actual[ind], _ = _play_turn(actual[ind],
                                            dice[strategies[ind](actual[ind])],
                                            board)
        winners = 0
        for act in actual:
            if act == board.goal:
                winners += 1
        for act, ind in zip(actual, range(len(actual))):
            if act == board.goal:
                results['wins'][ind] += 1./winners
        if winners > 1:
            results['equal'] += 1
    return results


if __name__ == "__main__":
    seed(789)
    traps = {'trap1': [1, 10], 'trap2': [3, 5, 7]}
    my_board = Board(traps, circling=False)
    n_games = 1000
    ###########
    # matrix = [0]*3
    # iter = 0
    # for my_die, x in play(my_board, n_games=n_games).items():
    #     matrix[iter] = x
    #     iter += 1
    #     print(str(my_die), " = ", str(x))
    #
    # strat = [-1]*my_board.goal
    # for i in range(my_board.goal):
    #     min_val = 1000000000
    #     min_die = -1
    #     for j in range(3):
    #         val = matrix[j][i]
    #         if val < min_val:
    #             min_val = val
    #             min_die = j
    #     strat[i] = min_die
    # print(str(strat))
    # print(play_strategy(my_board, lambda _x: strat[_x], n_games=n_games))
    #############################
    Expec, Dice = markovDecision(my_board)
    mdp_dice = [d-1 for d in Dice]
    print(mdp_dice)
    # print(Expec)
    print(play_strategy(my_board, lambda _x: mdp_dice[_x], n_games=n_games))
    # print("random for fun")
    # print(play_strategy(my_board, lambda _x: randrange(3), n_games=n_games))
    print("maybe intelligent")
    ai_dice = find_strategy(my_board, n_games=n_games, very_intelligent=True)
    print(ai_dice)
    print(play_strategy(my_board, lambda _x: ai_dice[_x], n_games=n_games))

    print("epic battles ensue: ")
    results = battle(my_board, [
        lambda _x: mdp_dice[_x],
        lambda _x: ai_dice[_x],
        lambda _x: randrange(3)
    ], n_games=10000)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(results)



