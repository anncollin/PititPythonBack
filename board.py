class Board : 
    def __init__(self,  trap_dic): 
        self.sart  = 0
        self.goal  = 14
        self.board = [1]*15 

        for _trap1 in trap_dic['trap1']: 
            self.board[_trap1] = 2
            
        for _trap2 in trap_dic['trap2']: 
            self.board[_trap2] = 3


    def successors(self, state):
        if (state == 2) : 
            return [3, 10]
        elif (state == 9) : 
            return [14]
        elif (state == 14) :
            return [0]
        else : 
            return [state+1]
        
       
