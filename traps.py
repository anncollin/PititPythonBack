"""
    Contains all the trap classes
"""


class Trap:
    __type = 0

    @staticmethod
    def get_lost_turns():
        return 0

    @staticmethod
    def get_new_state(state, board):
        return state


class TrapRestart(Trap):
    __type = 1

    @staticmethod
    def get_lost_turns():
        return 0

    @staticmethod
    def get_new_state(state, board):
        return 0


class TrapBack(Trap):
    __type = 2

    @staticmethod
    def get_lost_turns():
        return 0

    @staticmethod
    def get_new_state(state, board):
        return board.get_precursor(state, 3)


class TrapBlock(Trap):
    __type = 3

    @staticmethod
    def get_lost_turns():
        return 1

    @staticmethod
    def get_new_state(state, board):
        return state


def get_tile_trap(tile_type=0):
    all_traps = [Trap, TrapRestart, TrapBack, TrapBlock]
    return all_traps[tile_type]
