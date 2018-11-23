#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             config.py
# Description:      Meta-parameters and various options
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


#----------------------------------------------------------------------#
# board size : actually cant be modified anymore in this bitboard representation
L = 7
H = 6

#----------------------------------------------------------------------#
# Enter how many CPUs you want to use.
CPUS = 40
max_iterations = 1000 #max number of iteration of self play reinforcement learning

#----------------------------------------------------------------------#
#MCTS parameters
SIM_NUMBER = 30
sim_number_defense = 30 #old option : you can help the second player by increasing its sim number (not sure if it works)
CPUCT = 1
# Temperature :
tau=1
# see readme :
favorlonggames = True
long_game_factor = 0.1

# Unlike other github repo I choose the true reward for terminal states and not the NN Q-value (change to True if you want change this):
use_nn_for_terminal = False
# This is an (old) option to force the program to take the win when there is one, or counter the lose. It is actually not needed since it is going to learn this anyway
use_counter_in_mcts_nn = False
# To navigate in the MCTS tree, it looks reasonnable to mask and renormalize the probabilities given by the neural network when the move is not legal
# it is actually not required since the NN does learn it by itself (see the probability going to zero at turn 6 for the full central column)
maskinmcts = False

#----------------------------------------------------------------------#
#NN architecture

net = 'resnet' #other allowed choice is 'densenet'
res_tower = 1 #number of residual block : 1 is quite enough here with 256 filters
convsize = 256 #number of filters
polfilters = 2 # 256 -> 2 filters when entering policy head (DeepMind's choice)
valfilters = 1 # 256 -> 1 filter when entering value head (DeepMind's choice)
usehiddenpol = False #you can add a dense layer in the policy head. Default : None
hiddensize = 256 # hidden dense layer's size in the value head

#----------------------------------------------------------------------#
#choice of optimizer : allowed are 'sgd' or 'adam'
optim = 'sgd'
sgd_lr = 0.001 #initial learning rate. Curiously enough the learning is catastrophic for higher learning rates (like 0.1 in DeepMind's paper).
# I don't know why, probably because the NN is not a deep one with only one residual block?

#adam_lr=0.01

#learning rate annealing (learning rate is divided by 2 every 30 succesfull improvements of the NN):
lrdecay = 2
annealing = 30

#----------------------------------------------------------------------#
# Neural Net training

use_cuda = True #if you have a GPU
momentum = 0.9
wdecay = 0.0001 #weight decay
EPOCHS = 4
MINIBATCH = 32
MAXMEMORY = MINIBATCH * 3000 #one iteration of 400 games typically creates 600-1000 batches : here we thus save the last 10-6 games or so
MAXBATCHNUMBER = 1000 #and we improve the NN by sample randomly in the last maxmemory batches
MINBATCHNUMBER = 64

#----------------------------------------------------------------------#
#self play options
dirichlet_for_self_play = True
alpha_dir  = 0.8
epsilon_dir = 0.2
selfplaygames = 400 #i'd recommend at least 64

# see main functions :
use_z_last = False
data_extension = True

# temperatures
tau_zero_self_play = 18 #play greedily after turn 18
tau_self_play = 1 #temperature

#----------------------------------------------------------------------#
# check improvement or not of the NN
tournamentloop = 2 # this number * CPUS is the number of game you play to check whether the NN has improved
threshold = 0.51 #+ 1/np.sqrt(12*tournamentloop*CPUS) # This says it must be greater than 0.5 up to 1 standard deviation. This is 0.532 for 80 games
sim_number_tournaments = 49
tau_zero_eval_new_nn = 1 #in this tournament we set this parameter to 1 : both player are greedy
tau_pv = 1 # temperature

alternplayer = True # if set to False, the new NN player always start (which is a clear bias -> default is True)

# do we want use self play data from previous iterations? unclear. Both work (False is faster)
useprevdata = False

#----------------------------------------------------------------------#
#MCTS checkpoint options and ELO ratings
use_counter_in_pure_mcts = False
printstatefreq = 1
checkpoint_frequency = 1

#----------------------------------------------------------------------#
# print particular values on specific states

def particular_states():
    list=[[[],[]],[[24],[]],[[24],[25]],
          [[24,26],[25]], [[24,26],[25,27]], [[24,26,28],[25,27]], [[24,26,28],[25,27, 29]]]
    return list

def getstate(i):
    list = particular_states()
    elem = list[i]
    yellowboard = 0
    redboard = 0
    yellow = elem[0]
    red = elem[1]
    for u in yellow:
        yellowboard += 2**u
    for v in red:
        redboard += 2**v
    playerturn = 1
    if len(yellow)!=len(red):
        playerturn = - playerturn

    return [yellowboard, redboard, playerturn]


# our default NN architecture is : (kernel 4*4 reduces the board from 6*7 to 5*6
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1            [-1, 256, 5, 6]          12,288
#        BatchNorm2d-2            [-1, 256, 5, 6]             512
#               ReLU-3            [-1, 256, 5, 6]               0
#             Conv2d-4            [-1, 256, 5, 6]         589,824
#        BatchNorm2d-5            [-1, 256, 5, 6]             512
#               ReLU-6            [-1, 256, 5, 6]               0
#             Conv2d-7            [-1, 256, 5, 6]         589,824
#        BatchNorm2d-8            [-1, 256, 5, 6]             512
#               ReLU-9            [-1, 256, 5, 6]               0
#        BasicBlock-10            [-1, 256, 5, 6]               0
#            Conv2d-11              [-1, 2, 5, 6]             512
#       BatchNorm2d-12              [-1, 2, 5, 6]               4
#              ReLU-13              [-1, 2, 5, 6]               0
#            Linear-14                    [-1, 7]             427
#           Softmax-15                    [-1, 7]               0
#            Conv2d-16              [-1, 1, 5, 6]             256
#       BatchNorm2d-17              [-1, 1, 5, 6]               2
#              ReLU-18              [-1, 1, 5, 6]               0
#            Linear-19                  [-1, 256]           7,936
#              ReLU-20                  [-1, 256]               0
#            Linear-21                    [-1, 1]             257
#              Tanh-22                    [-1, 1]               0
# ================================================================
# Total params: 1,202,866
# Trainable params: 1,202,866
# Non-trainable params: 0
#
