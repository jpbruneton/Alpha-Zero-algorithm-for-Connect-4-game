#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             main_functions.py
# Description:      Main functions of the program, including self play,
#                   experience replay, tournaments and elo ratings
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #

# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
from MCTS_NN import MCTS_NN
from MCTS import MCTS
import random
from ResNet import ResNet_Training, DenseNet_Training
from Game_bitboard import Game
import config
import time
import pickle
from multiprocessing import Process
import matplotlib.pyplot as plt
import tqdm
import math
import os
import ResNet
import torch


# ======================= Useful fonctions for MAIN ========================== #

# --------------------------------------#
def load_or_create_neural_net():
    if config.net == 'resnet':
        file_path = './best_model_resnet.pth'
    elif config.net == 'densenet':
        file_path = './best_model_densenet.pth'
    else:
        print('Neural Net type not understood. Chose between resnet or densenet in config.py')
        raise ValueError

    if config.net == 'resnet':
        if os.path.exists(file_path):
            print('loading already trained model')
            time.sleep(0.3)

            best_player_so_far = ResNet.resnet18()
            best_player_so_far.load_state_dict(torch.load(file_path))
            best_player_so_far.eval()

        else:
            print('Trained model doesnt exist. Starting from scratch.')
            time.sleep(0.3)

            best_player_so_far = ResNet.resnet18()
            best_player_so_far.eval()

    if config.net == 'densenet':
        if os.path.exists(file_path):
            print('loading already trained model')
            time.sleep(0.3)

            best_player_so_far = ResNet.densenet()
            best_player_so_far.load_state_dict(torch.load(file_path))
            best_player_so_far.eval()

        else:
            print('Trained model doesnt exist. Starting from scratch.')
            time.sleep(0.3)
            best_player_so_far = ResNet.densenet()
            best_player_so_far.eval()

    return best_player_so_far


# ---------------------------------------------------------------------------- #
# Neural Net training
def improve_model_resnet(player, data, i):
    #here i is the number of times NN has improved : it will be used for learning rate annealing

    min_data=config.MINIBATCH* config.MINBATCHNUMBER
    max_data=config.MINIBATCH * config.MAXBATCHNUMBER
    size_training_set = data.shape[0]

    if size_training_set >= min_data:
        print('size of train set', size_training_set)

        #random sample from past experiences :
        if size_training_set >= max_data:
            X = data[np.random.choice(data.shape[0], max_data, replace=False)]
            #X = data[-size_training_set:-1,:]
        else:
            X=data

        # learning rate annealing: decrease by config.lrdecay the learning rate every config.annealing succesfull improvements of the model
        j= i//config.annealing

        if config.optim == 'sgd':
            lr_decay =config.sgd_lr/((config.lrdecay)**j)
        elif config.optim == 'adam':
            lr_decay = config.adam_lr / ((config.lrdecay) ** j)

        k = (i - 1) // config.annealing
        if j == k + 1:
            print('learning rate is now = ', lr_decay)

        if config.net == 'resnet':
            training = ResNet_Training(player,config.MINIBATCH,config.EPOCHS,lr_decay,X,X,1)
            training.trainNet()

        if config.net == 'densenet':
            training = DenseNet_Training(player, config.MINIBATCH, config.EPOCHS, lr_decay, X, X, 1)
            training.trainNet()

    else:
        print('Not enough training data. Please increase number of self play games or use previous data = True')
        print('train set size', data.shape[0])
        time.sleep(.1)
        raise ValueError


# ---------------------------------------------------------------------------- #
# play *one* game between two NN players but budget = number of sims
def onevsonegame(player1, budget1, player2, budget2, whostarts, cpuct, tau, tau_zero, use_dirichlet, index):

    #not sure if required but safety first!
    random.seed()
    np.random.seed()

    new_data_for_the_game = np.zeros((3*config.L*config.H + config.L + 1))

    if whostarts == 'player1':
        modulo = 1
        budget1=config.SIM_NUMBER
        budget2=config.sim_number_defense

    elif whostarts == 'player2':
        modulo = 0
        budget2 = config.SIM_NUMBER
        budget1 = config.sim_number_defense

    gameover = 0
    turn = 0

    while gameover == 0:
        turn = turn + 1

        if turn % 2 == modulo:
            player = 'player1'
            sim_number = budget1
            who_plays = player1
        else:
            player = 'player2'
            sim_number = budget2
            who_plays = player2

        #init tree
        if turn == 1:
            game = Game()
            tree = MCTS_NN(who_plays, use_dirichlet)
            rootnode = tree.createNode(game.state)
            currentnode = rootnode

        for sims in range(0, sim_number):
            tree.simulate(currentnode, cpuct)

        visits_after_all_simulations = []
        childmoves=[]

        for child in currentnode.children:
            visits_after_all_simulations.append(child.N**(1/tau))
            childmoves.append(child.move)

        all_visits=np.asarray(visits_after_all_simulations)
        probvisit = all_visits / np.sum(all_visits)
        child_col = [game.convert_move_to_col_index(move) for move in childmoves]

        #store the data created
        child_col = np.asarray(child_col, dtype=int)
        unmask_pi = np.zeros(config.L)
        unmask_pi[child_col] = probvisit
        flatten_state = game.state_flattener(currentnode.state)

        #init z to zero ; z is the actual reward from the current's player point of view, see below
        this_turn_data = np.hstack((flatten_state, unmask_pi,0))
        new_data_for_the_game = np.vstack((new_data_for_the_game, this_turn_data))

        #then take a step
        if turn < tau_zero:
            currentnode = np.random.choice(currentnode.children, p=probvisit)
        else:
            max = np.random.choice(np.where(all_visits == np.max(all_visits))[0])
            currentnode = currentnode.children[max]

        # reinit tree for next turn
        game = Game(currentnode.state)
        if player=='player1':
            tree = MCTS_NN(player2,use_dirichlet)
        else:
            tree = MCTS_NN(player1, use_dirichlet)

        rootnode = tree.createNode(game.state)
        currentnode = rootnode

        gameover = currentnode.isterminal()

    # game has terminated. Then, exit while, and  :
    new_data_for_the_game = np.delete(new_data_for_the_game, 0, 0)

    game = Game(currentnode.state)
    gameover, winner = game.gameover()

    if config.use_z_last:
        #include last winning move? unclear because there we don't have probabilities => put uniform prob
        # default : don't use z_last
        flatten_state = game.state_flattener(currentnode.state)
        unmask_pi = np.ones(config.L) / config.L
        this_turn_data = np.hstack((flatten_state, unmask_pi, 0))
        new_data_for_the_game = np.vstack((new_data_for_the_game, this_turn_data))

    #update the z's and winner stats
    wp1 = 0 # win player 1, etc
    wp2 = 0
    winstart=0
    winsecond=0
    draw = 0

    # backfill the z such as it becomes the actual reward from the current's player point of view:
    history_size = new_data_for_the_game.shape[0]

    if winner == 0:
        z = 0
        draw = 1

    elif winner == 1:
        if config.favorlonggames:
            z = 1 - config.long_game_factor*history_size/42 #the reward is bigger for shorter games
        else:
            z = 1
        winstart+=1
        if whostarts == 'player1':
            wp1 = 1
        else:
            wp2 = 1

    elif winner == -1:
        winsecond += 1
        if config.favorlonggames:
            z = -1 + config.long_game_factor*history_size/42 #the reward is less negative for long games
        else :
            z = -1
        if whostarts == 'player1':
            wp2 = 1
        else:
            wp1 = 1

    z_vec = np.zeros(history_size)

    for i in range(history_size):
        z_vec[i] = ((-1)**i)*z

    new_data_for_the_game[:, -1] = z_vec

    #data extension using parity along the x axis
    board_size=config.L*config.H

    if config.data_extension:
        extend_data=np.zeros((new_data_for_the_game.shape[0], new_data_for_the_game.shape[1]))

        for i in range(extend_data.shape[0]):
            board=np.copy(new_data_for_the_game[i, 0:3*board_size]).reshape((3, config.H,config.L))
            yellowboard=board[0]
            redboard=board[1]
            player_turn=board[2]

            #parity operation on array for both yellow and red boards
            flip_yellow=np.fliplr(yellowboard)
            flip_red=np.fliplr(redboard)

            extend_data[i, 0:board_size]= flip_yellow.flatten()
            extend_data[i, board_size:2*board_size]= flip_red.flatten()
            extend_data[i, 2*board_size: 3*board_size] = player_turn.flatten()

            # parity operation on the Pi's
            pi_s= np.copy(new_data_for_the_game[i, 3*board_size:3*board_size + config.L])
            flip_pi=np.flip(pi_s, axis=0)

            extend_data[i, 3*board_size:3*board_size + config.L] = flip_pi
            extend_data[i, -1] = np.copy(new_data_for_the_game[i,-1])

        #stack
        new_data_for_the_game = np.vstack((new_data_for_the_game, extend_data))

    #save data of self play in a file indexed by the CPU used.
    mydata={'data' : [new_data_for_the_game, wp1,wp2, draw,winstart,winsecond, history_size]}
    filename = './data/createdata' + str(index) + '.txt'
    with open(filename, 'wb') as file:
        pickle.dump(mydata, file)
    file.close()

# ---------------------------------------------------------------------------- #
# main self play function

def self_play(player, self_play_loop_number, CPUs, sim_number, cpuct, tau, tau_zero, use_dirichlet):
    winp1 = 0
    winp2 = 0
    draws = 0
    w_player_start = 0
    w_second_player = 0

    new_data = np.zeros((3*config.L * config.H + config.L + 1))

    for _ in tqdm.tqdm(range(self_play_loop_number)):

        #parallelize
        procs = []

        for index in range(CPUs):
            if index % 2 == 0:
                whostarts = 'player1'
            else:
                whostarts = 'player2'

            proc = Process(target=onevsonegame,
                           args=(player, sim_number, player, sim_number,
                                 whostarts, cpuct, tau, tau_zero, use_dirichlet, index,))
            procs.append(proc)

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        #end of parallel self play games. Retrieve data :
        for index in range(CPUs):
            filename = './data/createdata' + str(index) + '.txt'
            with open(filename, 'rb') as file:
                load_dic = pickle.load(file)

            get_data, wp1, wp2, draw, winstart, winsecond , history_size = load_dic['data']

            winp1 += wp1
            winp2 += wp2
            draws += draw
            w_player_start += winstart
            w_second_player += winsecond

            new_data = np.vstack((new_data, get_data))
            file.close()

    if w_player_start + w_second_player==0:
        ratio = 0
    else:
        ratio = w_player_start/(w_player_start + draws+ w_second_player)

    new_data = np.delete(new_data, 0, 0)

    return new_data, winp1, winp2, draws, ratio

# ---------------------------------------------------------------------------- #
# main tournament function between version1 NN and version 2 NN

def play_v1_against_v2(current_player, best_player_so_far,
                       loop_number, CPUs, sim_number, cpuct, tau, tau_zero, use_dirichlet):
    winp1 = 0
    winp2 = 0
    draws = 0
    w_first = 0
    w_second = 0


    for _ in tqdm.tqdm(range(loop_number)):

        procs = []

        #if alternplayer is true, player 1 starts half of the games, and player 2 the other half
        for index in range(CPUs):
            if index % 2 == 0:
                whostarts = 'player1'
            else:
                if config.alternplayer:
                    whostarts = 'player2'
                else:
                    whostarts = 'player1'

            #here player 1 is the improved NN, player 2 the old NN
            proc = Process(target=onevsonegame,
                           args=(current_player, sim_number, best_player_so_far, sim_number,
                                 whostarts, cpuct, tau, tau_zero, use_dirichlet, index,))
            procs.append(proc)

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        #end of games.
        for index in range(CPUs):
            filename = './data/createdata' + str(index) + '.txt'
            with open(filename, 'rb') as file:
                load_dic = pickle.load(file)

            new_data, wp1, wp2, draw, wf, ws, history_size = load_dic['data']
            w_first += wf
            w_second += ws
            winp1 += wp1
            winp2 += wp2
            draws += draw



    if w_first + w_second==0:
        ratio = 0
    else:
        ratio = w_first/(w_first+draws+w_second)

    return winp1, winp2, draws, ratio

# --------------------------------------------------------------------#

def generate_self_play_data(best_player_so_far, sim_number, dataseen, i):
    print('')
    print('--- Generating data with self-play (', (config.selfplaygames // config.CPUS) * config.CPUS, 'games) ---',
          'iteration number', i)
    time.sleep(0.01)

    # if so, we stack together the previous data with the new one
    if config.useprevdata:
        local_data, winp1, winp2, draws, ratio = \
            self_play(best_player_so_far, config.selfplaygames // config.CPUS, config.CPUS,
                                     sim_number, config.CPUCT, config.tau_self_play,
                                     config.tau_zero_self_play, config.dirichlet_for_self_play)
        time.sleep(0.01)
        print('FYI, win ratio of first player was', int(ratio * 1000) / 10, '%')
        time.sleep(0.01)

        prev_data_seen = np.copy(dataseen)
        use_this_data = np.vstack((dataseen, local_data))

        #and delete the init
        if np.sum(np.abs(use_this_data[0, :])) == 0:
            use_this_data = np.delete(use_this_data, 0, 0)

    # if we don't. Default config is : we don't
    else:
        use_this_data, winp1, winp2, draws, ratio = \
            self_play(best_player_so_far, config.selfplaygames // config.CPUS, config.CPUS,
                                     sim_number, config.CPUCT, config.tau_self_play,
                                     config.tau_zero_self_play, config.dirichlet_for_self_play)
        #not used but necessary
        prev_data_seen = np.copy(dataseen)

        time.sleep(0.01)
        print('FYI, win ratio of first player was', int(ratio * 1000) / 10, '%')
        time.sleep(0.01)

    return use_this_data, prev_data_seen




# -----------------------------------------------------------------------#
# UCT evaluator for pure MCTS
def UCT_simu(node, Cp):
    if node.N == 0:
        return 1000
    else:
        return node.Q + Cp * np.sqrt(2 * np.log(node.parent.N) / (node.N))

# -----------------------------------------------------------------------#
# play *one* game between NN and pure MCTS

def NN_against_mcts(player_NN, budget_NN, budget_MCTS, whostarts, c_uct, cpuct, tau, tau_zero, use_dirichlet, index):
    random.seed()
    np.random.seed()

    if whostarts == 'player_nn':
        modulo = 1
    elif whostarts == 'player_mcts':
        modulo = 0


    w_nn_start = 0
    w_nn_second = 0
    gameover = 0
    turn = 0

    while gameover == 0:

        turn = turn + 1

        if turn % 2 == modulo:
            player = 'player_nn'
            sim_number = budget_NN

        else:
            player = 'player_mcts'
            sim_number = budget_MCTS

        #init tree for NN or MCTS
        if turn == 1:
            if player == 'player_nn':
                game = Game()
                tree = MCTS_NN(player_NN, use_dirichlet)
                rootnode = tree.createNode(game.state)
                currentnode = rootnode
            else:
                game = Game()
                tree = MCTS()
                rootnode = tree.createNode(game.state)
                currentnode = rootnode

        if player=='player_nn':

            for sims in range(0, sim_number):
                tree.simulate(currentnode, cpuct)

            visits_after_all_simulations = []

            for child in currentnode.children:
                visits_after_all_simulations.append(child.N**(1/tau))

            all_visits=np.asarray(visits_after_all_simulations)
            probvisit = all_visits / np.sum(all_visits)

            # take a step
            if turn < tau_zero:
                currentnode = np.random.choice(currentnode.children, p=probvisit)
            else:
                max = np.random.choice(np.where(all_visits == np.max(all_visits))[0])
                currentnode = currentnode.children[max]

            # reinit tree for next player : mcts
            game = Game(currentnode.state)
            tree = MCTS()
            rootnode = tree.createNode(game.state)
            currentnode = rootnode
            gameover = currentnode.isterminal()

        if player=='player_mcts':
            for sims in range(0, sim_number):
                tree.simulate(currentnode, UCT_simu, c_uct, config.use_counter_in_pure_mcts)

            visits_after_all_simulations = []

            for child in currentnode.children:
                visits_after_all_simulations.append(child.N)

            values = np.asarray(visits_after_all_simulations)
            imax = np.random.choice(np.where(values == np.max(values))[0])
            currentnode = currentnode.children[imax]

            # reinit tree for next player : neural net
            game = Game(currentnode.state)
            tree = MCTS_NN(player_NN, use_dirichlet)
            rootnode = tree.createNode(game.state)
            currentnode = rootnode
            gameover = currentnode.isterminal()

    game = Game(currentnode.state)
    gameover, winner = game.gameover()

    wp1 = 0
    wp2 = 0
    draw = 0

    if winner == 0:
        draw = 1

    elif winner == 1:
        if whostarts == 'player_nn':
            wp1 = 1
            w_nn_start = 1

        else:
            wp2 = 1

    elif winner == -1:
        if whostarts == 'player_nn':
            wp2 = 1
        else:
            wp1 = 1
            w_nn_second = 1

    save_dic = {}
    save_dic['data'] = np.asarray([wp1, wp2, draw,w_nn_start,w_nn_second])
    filename = './data/nn_against_mcts' + str(index) + '.txt'
    with open(filename, 'wb') as file:
        pickle.dump(save_dic, file)
    file.close()


# ---------------------------------------------------------------------------- #
# Use this as a checkpoint for later use to compute the elo rating
# Here we play parallel games of NN against pure MCTS

def winrate_against_mcts(player, sim_number, self_play_loop_number,
                         CPUs, budget_mcts, cpuct, tau, tau_zero, use_dirichlet):
    winp1 = 0
    winp2 = 0
    draws = 0
    w_nn_start=0
    w_nn_second=0
    c_uct = config.CPUCT

    for _ in range(self_play_loop_number):

        procs = []

        for index in range(CPUs):
            if index % 2 == 0:
                whostarts = 'player_nn'
            else:
                whostarts = 'player_mcts'

            proc = Process(target=NN_against_mcts,
                           args=(player, sim_number, budget_mcts,
                                 whostarts, c_uct, cpuct, tau, tau_zero, use_dirichlet, index,))
            procs.append(proc)

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        # end of games

        for index in range(CPUs):
            filename = './data/nn_against_mcts' + str(index) + '.txt'
            with open(filename, 'rb') as file:
                load_dic = pickle.load(file)

            wp1, wp2, draw,  w_start, w_second = load_dic['data']

            winp1 += wp1
            winp2 += wp2
            draws += draw
            w_nn_start+= w_start
            w_nn_second += w_second
            file.close()
    if winp1 == 0:
        ratio_starter = 0
    else:
        ratio_starter = w_nn_start/winp1
    return winp1, winp2, draws, ratio_starter


#---------------------------------------------------------------------#
def geteloratings(elos, best_player_so_far, improved, total_improved):

    # increase slowly the strenght of the mcts we play against (otherwise you soon get 100% wins against 100 sims-mcts and elo cant be computed anymore
    # numbers from pre_compute_elo_ratings/draw_elo.py
    if total_improved < 15 :
        budget_mcts = 100
        baserating = 920
    elif total_improved < 30:
        budget_mcts = 200
        baserating = 1057
    elif total_improved < 60:
        budget_mcts = 400
        baserating = 1184
    elif total_improved < 90:
        budget_mcts = 800
        baserating = 1286
    elif total_improved < 120:
        budget_mcts = 1600
        baserating = 1392
    else :
        budget_mcts = 3200
        baserating = 1495

    use_dirichlet = False #greedy NN player
    tau_agg = 1
    tau_zero = 1 #greedy NN player
    loop_number_mcts = 2   #meaning we play 2*cpus games
    sim_number_a_mcts = 49 #sim number allowed for the trained neural net

    if improved == 1 and total_improved % config.checkpoint_frequency == config.checkpoint_frequency-1:
        print('')
        print('---CHECKPOINT: winrate against mcts with', budget_mcts, 'sims (', config.CPUS*loop_number_mcts, 'games)')
        time.sleep(0.01)

        winp1, winp2, draws, ratio_starter = \
            winrate_against_mcts\
                (best_player_so_far,sim_number_a_mcts, loop_number_mcts,
                 config.CPUS, budget_mcts, config.CPUCT, tau_agg, tau_zero,use_dirichlet)

        print('NN wins by', 100*winp1/(winp1 + winp2 + draws), 'draw', 100*draws/(winp1 + winp2 + draws), 'lost', 100*winp2/(winp1 + winp2 + draws) )
        time.sleep(0.01)
        print('when NN wins, it wins', 100*ratio_starter, '% of the won games as first player' )

        #in case there is a 0% or 100% win - decide its +-800 points in Elo
        #doesnt happen often with the increasing mcts budget

        eloscore = (winp1 + draws/2)/(winp1 + winp2 + draws)
        if eloscore == 1 :
            eloscore = 0.99
        if eloscore == 0:
            eloscore = 0.01

        nn_elo_rating = baserating - 400*math.log(1/eloscore -1,10)
        print('estimated ELO rating is', int(nn_elo_rating))
        elos.append(int(nn_elo_rating))
        print(elos)
        time.sleep(0.01)

    return elos


# -----------------------------------------------------------------------------------------------------#
# display the evolution of probabilities for particularly important states (at the beginning of the game)
# see also readme file

def printstates(player):
    part_states = config.particular_states()
    # knowledge based on http://connect4.gamesolver.org
    # for instance for turn 5 : http://connect4.gamesolver.org/?pos=44444
    print('')
    print('(probs should be max for optimal play, ie [0,0,0,1,0,0,0] from turn 0 to 4 included ; turn 5 flat prob, turn 6, [0,0,.5,0,0.5,0,0]')
    print('Q-values of the board should be minimal for the corresponding optimal move)')
    print('')
    getbreak=1

    for i in range(len(part_states)):

        state = config.getstate(i)
        game = Game(state)

        dirichletforprinting=False
        tree = MCTS_NN(player, dirichletforprinting)
        rootnode = tree.createNode(game.state)
        tree.expand_all(rootnode)
        tree.eval_leaf(rootnode)
        pchild = rootnode.proba_children
        pchild = [int(1000 * x) / 10 for x in pchild]


        for child in rootnode.children:
            tree.expand_all(child)
            tree.eval_leaf(child)

        Qs = [- int(100*child.Q)/100 for child in rootnode.children]
        Qchilds=[-child.Q for child in rootnode.children]

        turn = str(bin(state[0])).count('1') +str(bin(state[1])).count('1')
        print('turn', int(turn), 'Qval of this board', - int(1000*rootnode.Q)/1000)
        print('children probs', pchild, 'and of corresponding Q-val', Qs)
        time.sleep(0.01)


        #for automatic break of the main loop when the model is good enough
        #we require probabilities for central column to be at least 92%
        if int(turn) <= 4 and pchild[3] < 92:
            getbreak = 0

        # and lowest Q-value for the optimal move
        if int(turn) <= 4 :
            if Qchilds[3] > Qchilds[0]  or Qchilds[3] > Qchilds[1] or Qchilds[3] > Qchilds[2] or Qchilds[3] > Qchilds[4] or Qchilds[3] > Qchilds[5] or Qchilds[3] > Qchilds[6]:
                getbreak = 0

        #and, in the main program, an ELO of at least 1800 (see main)
    return getbreak
