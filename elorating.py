#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             elorating.py
# Description:      This allows to directly compute winrate and elorating of a given best_model against given MCTS sim_number
#                   Almost identical/taken from https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #

import config
import main_functions
from ResNet import ResNet
import ResNet
import torch.utils


def launch():
    sim_number_a_mcts = 200
    self_play_loop_number_mcts = 4 #playing 4*cpus games
    CPUs = config.CPUS
    cpuct = 1
    tau_agg = 1
    tau_zero = 1
    use_dirichlet = False

    file_path_resnet = './best_model_resnet.pth'
    best_player_so_far = ResNet.resnet18()
    best_player_so_far.load_state_dict(torch.load(file_path_resnet))

    budgets = [1000, 10000, 50000] #these are quite long runs : NN is fast, but MCTS with so many sims is slow
    for x in budgets:
        budget_mcts = x
        winp1, winp2, draws, _ = main_functions.winrate_against_mcts(best_player_so_far,sim_number_a_mcts, self_play_loop_number_mcts,CPUs, budget_mcts, cpuct, tau_agg, tau_zero,use_dirichlet)

        print('NN against MCTS with', budget_mcts, 'sims', 'win :', 100*winp1/(winp1 + winp2 + draws), 'draw', 100*draws/(winp1 + winp2 + draws), 'lost', 100*winp2/(winp1 + winp2 + draws) )



if __name__ == '__main__':
    launch()