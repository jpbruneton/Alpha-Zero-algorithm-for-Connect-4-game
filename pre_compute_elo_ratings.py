#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             pre_compute_elo_ratings.py
# Description:      we make tournaments against pure mcts to determine an ELO scale, given some origin
#                   we set random player ELO rating to 0
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
from MCTS import MCTS
import numpy as np
from Game_bitboard import Game
import pickle
from multiprocessing import Process
import random
import config
import tqdm
import math

# --------------------------------------------------------------------- #
def UCT_simu(node, Cp):
    if node.N == 0:
        return 1000
    else:
        return node.Q + Cp * np.sqrt(2 * np.log(node.parent.N) / (node.N))

# --------------------------------------------------------------------- #
def onevsonegame(budget1, random1, counter1,  usecounter_in_rollout_1, budget2, random2, counter2, usecounter_in_rollout_2, whostarts, index):

    import random
    random.seed()
    np.random.seed()

    if whostarts == 'budget1':
        modulo = 1
    elif whostarts == 'budget2':
        modulo = 0

    # init tree, root, game
    tree = MCTS()
    c_uct = 1
    game = Game()
    turn = 0
    gameover = 0
    rootnode = tree.createNode(game.state)
    currentnode = rootnode

    # main loop
    while gameover == 0:

        turn = turn + 1

        if turn % 2 == modulo:
            #player = 'player1'
            sim_number = budget1
            usecounterinrollout=usecounter_in_rollout_1
            counter=counter1
            rd=random1

        else:
            #player = 'player2'
            sim_number = budget2
            usecounterinrollout=usecounter_in_rollout_2
            counter=counter2
            rd=random2

        if rd: #completely random play / or random + counter
            if counter:
                currentnode, existscounter = getcountermove(currentnode, tree)
                if existscounter == False:
                    if len(currentnode.children) == 0:
                        tree.expand_all(currentnode)
                    randindex = int(random.random() * (len(currentnode.children)))
                    currentnode = currentnode.children[randindex]

            else:
                if len(currentnode.children) == 0:
                    tree.expand_all(currentnode)
                randindex = int(random.random() * (len(currentnode.children)))
                currentnode = currentnode.children[randindex]

        else:
            if counter:
                currentnode, existscounter = getcountermove(currentnode, tree)
                if existscounter == False:
                    for sims in range(0, sim_number):
                        tree.simulate(currentnode, UCT_simu, c_uct, usecounterinrollout)

                    visits = np.array([child.N for child in currentnode.children])
                    max_visits = np.where(visits == np.max(visits))[0]
                    imax = max_visits[int(random.random() * len(max_visits))]
                    currentnode = currentnode.children[imax]

            else:

                for sims in range(0, sim_number):
                    tree.simulate(currentnode, UCT_simu, c_uct, usecounterinrollout)

                visits = np.array([child.N for child in currentnode.children])
                max_visits = np.where(visits == np.max(visits))[0]
                imax = max_visits[int(random.random() * len(max_visits))]
                currentnode = currentnode.children[imax]

        # then reinit tree
        game = Game(currentnode.state)
        tree = MCTS()
        rootnode = tree.createNode(game.state)
        currentnode = rootnode
        gameover, winner = game.gameover()

    #print('end of game')
    if winner == 0:
        toreturn = 'draw'

    elif winner == 1:
        if whostarts == 'budget1':
            toreturn = 'budget1'
        else:
            toreturn = 'budget2'

    elif winner == -1:
        if whostarts == 'budget1':
            toreturn = 'budget2'
        else:
            toreturn = 'budget1'

    monresult={'result' : toreturn}
    filename = './data/game' + str(index) + '.txt'
    with open(filename, 'wb') as file:
        pickle.dump(monresult, file)
    file.close()

def getcountermove(currentnode, tree):
    existcounter=False
    game = Game(currentnode.state)
    can_win, where_win, can_be_lost, where_lose = game.iscritical()

    if can_win == 1: #then take it
        move = where_win[int(random.random() * len(where_win))]
        tree.expand_all(currentnode)  # must expand since not done in mcts sims in that case
        col = game.convert_move_to_col_index(move)
        for child in currentnode.children:
            child_col = game.convert_move_to_col_index(child.move)
            if child_col == col:
                currentnode = child
        existcounter = True

    elif can_be_lost == 1: # then counter
        move = where_lose[int(random.random() * len(where_lose))]
        tree.expand_all(currentnode)  # must expand since not done in mcts
        col = game.convert_move_to_col_index(move)
        for child in currentnode.children:
            child_col = game.convert_move_to_col_index(child.move)
            if child_col == col:
                currentnode = child
        existcounter = True

    return currentnode, existcounter


def tournaments(budget1, random1, counter1,  usecounter_in_rollout_1, budget2, random2,
                      counter2, usecounter_in_rollout_2, loop_number):

    np.random.seed()
    random.seed()

    win_b1 = 0
    win_b2 = 0
    draws = 0
    tot_games = 0

    for i in tqdm.tqdm(range(loop_number)):
        procs = []
        for i in range(config.CPUS):
            if i % 2 == 0:
                whostarts = 'budget1'
            else:
                whostarts = 'budget2'
            proc = Process(target=onevsonegame, args=(budget1, random1, counter1, usecounter_in_rollout_1,
                                                           budget2, random2, counter2,
                                  usecounter_in_rollout_2, whostarts, i))
            procs.append(proc)

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        for i in range(config.CPUS):
            filename = './data/game' + str(i) + '.txt'
            with open(filename, 'rb') as file:
                load_dic = pickle.load(file)

            result = load_dic['result']

            if result == 'budget2':
                win_b2 += 1
            elif result == 'draw':
                draws += 1
            else:
                win_b1 += 1

            tot_games += 1

    print('end of tournament with', tot_games, 'games played')
    print('and results', 'p1 win rate :', 100 * win_b1 / tot_games, 'draws :', 100 * draws / (tot_games),
          'player 2 win rate: ', 100 * win_b2 / (tot_games))
    print('player1 score', 100 * win_b1 / tot_games + 100 * draws / (2 * tot_games))
    score = win_b1 / tot_games + draws / (2 * tot_games)

    return 100 * win_b1 / tot_games, 100 * draws / (tot_games), 100 * win_b2 / (tot_games), score


def launch():

    results=[]
    #enter here what you want to play
    budgets=[[10000, 3200], [12800, 6400], [50000, 12800]]
    for x in budgets:
        budget1 = x[0]
        random1 = False
        counter1 = False
        usecounter_in_rollout_1 = False
        budget2 = x[1]
        random2 = False
        counter2 = False
        usecounter_in_rollout_2 = False
        loop_number = 10 # it is going to play loop number * cpus game to determine the elo

        p1wr, draws, p2wr, score = tournaments(budget1, random1, counter1,  usecounter_in_rollout_1, budget2, random2,
                          counter2, usecounter_in_rollout_2, loop_number)

        # format [simu, score]
        deltaelo= - 400 * math.log(1/score - 1, 10)
        results.append([budget1, budget2, deltaelo])
        print(results)

if __name__ == '__main__':
    launch()

#after many runs:
#all based on 20000 games. ELO are then accurate +- 15 points
# remember : elo rating of random player is 0. For info, elo rating of random player + take win/counter lose is 565
# (format : sim number of mcts, elo rating)

#results1 = [[10,250],[20, 500],[30, 603],[40, 670],[50,736],[60, 783],[70,822], [80, 860], [90, 890], [100, 920], [[200, 1057]

#all based on 5000 games.
#results2 = [[400, 1184 ],[800, 1286 ],[1600, 1392 ]]

#all based on 1600 games.
#results2 = [[3200,  ],[6400,  ],]




