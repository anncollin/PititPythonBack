def markovDecision(board):
    Expec             = [1]*15 
    Expec[board.goal] = 0
    Dice              = [0]*15 

    # value-iteration algorithm
    iter = 0
    all_expect = [[0]*3]*15
    variance = 1
    while variance > 0.00003 : # Normally until convergence...
        for i in range(0,14): 
            successors = board.successors(i)

            # Expected value when security dice is used 
            exp_security = 1 + 0.5 * Expec[i] # Cost of 1 action + do not move
            for succ in successors: # move of 1
                exp_security+= (0.5/len(successors)) * Expec[succ]  


            # Expected value when normal dice is used 
            exp_normal = 1
            if board.board[i] == 2 : #Trap 1
                exp_normal+= (1/3) * (0.5 * Expec[i] + 0.5 * Expec[0]) 
            elif board.board[i] == 3 : #Trap 2
                exp_normal+= (1/3) * (0.5 * Expec[i] + 0.5 * Expec[(i-3)%15])
            else : #No trap 
                exp_normal+= (1/3) * Expec[i]
            
            for succ in successors: 
                if board.board[succ] == 2 : #Trap 1
                    exp_normal+= ((1/3)/len(successors)) * (0.5 * Expec[succ] + 0.5 * Expec[succ]) 
                elif board.board[i] == 3 : #Trap 2
                    exp_normal+= ((1/3)/len(successors)) * (0.5 * Expec[succ] + 0.5 * Expec[(succ-3)%15]) 
                else : #No trap 
                    exp_normal+= ((1/3)/len(successors)) * Expec[succ]

                successors2 = board.successors(succ)
                for succ2 in successors2: 
                    if board.board[succ2] == 2 : #Trap 1
                        exp_normal+= ((1/3)/len(successors2)) * (0.5 * Expec[succ2] + 0.5 * Expec[succ2]) 
                    elif board.board[i] == 3 : #Trap 2
                        exp_normal+= ((1/3)/len(successors2)) * (0.5 * Expec[succ2] + 0.5 * Expec[(succ2-3)%15]) 
                    else : #No trap 
                        exp_normal+= ((1/3)/len(successors2)) * Expec[succ2]


            # Expected value when risky dice is used 
            exp_risky = 1 

            if board.board[i] == 2 : #Trap 1
                exp_risky+= (1/4) * (0.5 * Expec[i] + 0.5 * Expec[0]) 
            elif board.board[i] == 3 : #Trap 2
                exp_risky+= (1/4) * (0.5 * Expec[i] + 0.5 * Expec[(i-3)%15])
            else : #No trap 
                exp_risky+= (1/4) * Expec[i]

            for succ in successors: 
                if board.board[succ] == 2 :#Trap 1
                    exp_risky+= ((1/4)/len(successors)) * Expec[0] 
                elif board.board[succ] == 3 :#Trap 2
                    exp_risky+= ((1/4)/len(successors)) * Expec[(succ-3)%15]
                else : 
                    exp_risky+= ((1/4)/len(successors)) * Expec[succ] 

                successors2 = board.successors(succ)
                for succ2 in successors2: 
                    if board.board[succ2] == 2 :#Trap 1
                        exp_risky+= ((1/4)/len(successors2)) * Expec[0] 
                    elif board.board[succ2] == 3 :#Trap 2
                        exp_risky+= ((1/4)/len(successors2)) * Expec[(succ2-3)%15] 
                    else : 
                        exp_risky+= ((1/3)/len(successors2)) * Expec[succ2]

                    successors3 = board.successors(succ2)
                    for succ3 in successors3: 
                        if board.board[succ3] == 2 :#Trap 1
                            exp_risky+= ((1/4)/len(successors3)) * Expec[0] 
                        elif board.board[succ] == 3 :#Trap 2
                            exp_risky+= ((1/4)/len(successors3)) * Expec[(succ3-3)%15] 
                        else : 
                            exp_risky+= ((1/4)/len(successors3)) * Expec[succ3]

            Expec[i] = min(exp_security, exp_normal, exp_risky)
            variance = abs(all_expect[i][0]-exp_security) \
                    + abs(all_expect[i][1]-exp_normal) \
                    + abs(all_expect[i][2]-exp_risky)
            all_expect[i] = [exp_security, exp_normal, exp_risky]
        iter += 1
    
    for j in range(0,15): 
        Dice[j] = all_expect[j].index(min(all_expect[j])) +1 
    return Expec, Dice