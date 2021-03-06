from dies import get_all_dice
from traps import get_tile_trap
from collections import defaultdict


# The Board class contains the board's informations
class Board:
    def __init__(self, trap_dic, circling=True):
        self.start = 0
        self.goal = 14
        self.circling = circling
        self.board = [get_tile_trap(0)] * 15

        for _trap in trap_dic['trap1']:
            self.board[_trap] = get_tile_trap(1)

        for _trap in trap_dic['trap2']:
            self.board[_trap] = get_tile_trap(2)

        for _trap in trap_dic['trap3']:
            self.board[_trap] = get_tile_trap(3)

        self.dice = get_all_dice()

    '''
        @state: the square which successors are wanted
        @first: true if the player starts his turn on @state
                false otherwise
         
        @return a list with the successors of @state         
    '''
    def get_successors(self, state, first=False):
        if state == 2 and first:
            return [3, 10]
        elif state == 2 and not first:
            return [3]
        elif state == 9:
            return [14]
        elif state == 14:
            if self.circling:
                return [0]
            else:
                return [14]
        else:
            return [state + 1]

    '''
        @state: the square which precursor is wanted
        @backtrack: the numbers of squares back

        @return state-backtrack applied to this board            
    '''
    def get_precursor(self, state, backtrack):
        if 10 <= state < 10 + backtrack:
            to_deduce = backtrack - (state-9)
            return max(0, 3 - to_deduce)
        return max(0, state - backtrack)

    def get_tile(self, index):
        return self.board[index]


# A conversion function to satisfy the project's signature
def make_board(arr):
    dico = defaultdict(list)
    for x, ind in zip(arr, range(len(arr))):
        if x != 0:
            if x > 3:
                print("Trap {} at index {} is not supported".format(x, ind))
                continue
            dico['trap{}'.format(x)] += [ind]

    return Board(dico)

