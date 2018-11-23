#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             play_against_human.py
# Description:      Allows you to play against the NN. You can start or let it start
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #
#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             play_against_human.py
# Description:      Allows you to play against a trained NN
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
from MCTS_NN import Node, MCTS_NN
from Game_bitboard import Game
import numpy as np
from ResNet import ResNet
import ResNet
import time
import torch.utils


def onevsonehuman(budget, whostarts):
    if whostarts == 'computer':
        modulo = 1
    else:
        modulo = 0

    file_path_resnet = './best_model_resnet.pth'
    best_player_so_far = ResNet.resnet18()
    best_player_so_far.load_state_dict(torch.load(file_path_resnet))

    game = Game()
    tree = MCTS_NN(best_player_so_far, use_dirichlet=False)
    rootnode = tree.createNode(game.state)
    currentnode = rootnode

    turn = 0
    isterminal = 0

    while isterminal == 0:

        turn = turn + 1

        if turn % 2 == modulo:
            player = 'computer'
            sim_number = budget
        else:
            player = 'human'

        if player=='computer':


            print('===============IA playing================')
            for sims in range(0, sim_number):
                tree.simulate(currentnode, cpuct=1)

            treefordisplay = MCTS_NN(best_player_so_far, False)
            rootnodedisplay = treefordisplay.createNode(game.state)
            treefordisplay.expand_all(rootnodedisplay)
            tree.eval_leaf(rootnodedisplay)
            pchild = rootnodedisplay.proba_children
            pchild = [int(1000 * x) / 10 for x in pchild]
            for child in rootnodedisplay.children:
                treefordisplay.eval_leaf(child)
            Qs = [int(100 * child.Q) / 100 for child in rootnodedisplay.children]
            print('NN thoughts', pchild, Qs)
            visits_after_all_simulations = []

            for child in currentnode.children:
                visits_after_all_simulations.append(child.N)

            print('result visits', visits_after_all_simulations)
            time.sleep(0.5)
            values = np.asarray(visits_after_all_simulations)
            imax = np.random.choice(np.where(values == np.max(values))[0])
            print('choice made', imax)
            currentnode = currentnode.children[imax]


        else: #human player
            print('=============== your turn =====================')
            game=Game(currentnode.state)
            game.display_it()
            moves=game.allowed_moves()
            print('chose a move from 0 to 6 -- beware of full columns! (not taken into account : e.g. if column three is full, enter 5 instead of 6 to play in the last column)')
            human_choice=int(input())
            game.takestep(moves[human_choice])
            currentnode=Node(game.state, moves[human_choice])

        # reinit tree
        game = Game(currentnode.state)
        tree = MCTS_NN(best_player_so_far, use_dirichlet=False)
        rootnode = tree.createNode(game.state)

        currentnode = rootnode

        isterminal = currentnode.isterminal()

    game = Game(currentnode.state)
    gameover, winner = game.gameover()

    #print('end of game')
    if winner == 0:
        toreturn = 'draw'
        print('draw')

    elif winner == 1:
        if whostarts == 'computer':
            print('computer wins')
            toreturn = 'budget1'

        else:
            print('you win')
            toreturn = 'budget2'

    elif winner == -1:
        if whostarts == 'computer':
            print(' you win')
            toreturn = 'budget2'

        else:
            print('computer wins')
            toreturn = 'budget1'


    return toreturn

#set the number of sims the NN player gets:
sim_number = 200
# set who starts, 'human' or 'computer'
onevsonehuman(200, 'human')
