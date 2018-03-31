

def _rec_exp(die, board, index, expec, stop):
    if stop == -1:
        return 0
    trap = board.get_tile(index)
    successors = board.get_successors(index, stop == die.adv)
    l_succ_div = 1./len(successors)

    ret = 1./(die.adv + 1.) * (
            (1-die.trap_prob) * expec[index] +
            die.trap_prob * (expec[trap.get_new_state(index, board)] + trap.get_lost_turns())
    )
    for succ in successors:
        ret += _rec_exp(die, board, succ, expec, stop-1) * l_succ_div

    return ret


def get_exp(die, board, index, expec):
    return 1 + _rec_exp(die, board, index, expec, die.adv)


# cfr @markovDecision function in main.py
def markov_decision(board):
    expec = [1] * 15
    expec[board.goal] = 0
    die_to_use = [board.dice[0]] * 15
    dice = board.dice

    # value-iteration algorithm
    _iter = 0
    all_expect = []
    for _ in range(15):
        all_expect += [[0] * 3]
    variance = 1
    while variance > 0.00003:  # This is very close to convergence
        variance = 0
        for i in range(14):
            exp_i = [0]*3
            for j in range(len(dice)):
                exp_i[j] = get_exp(dice[j], board, i, expec)
                variance += abs(all_expect[i][j] - exp_i[j])
                all_expect[i][j] = exp_i[j]

            expec[i] = min(all_expect[i])
        _iter += 1

    for j in range(0, 15):
        die_to_use[j] = all_expect[j].index(min(all_expect[j])) + 1
    return expec, die_to_use
