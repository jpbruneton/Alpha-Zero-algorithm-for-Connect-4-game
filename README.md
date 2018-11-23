# Alpha-Zero-algorithm-for-Connect-4-game
Implementation of Alpha Zero for Go of DeepMind for Connect 4 game in Python 3.6 and PyTorch

## Includes

- A bitboard encoding of the game environment - meaning it is specific to the 6*7 board

- Self-play games are parallelized on user's CPU's

- Model training on GPU if available

- Model can be a fully dense network with two dense heads, or a residual network with two heads. By default, it is a one block ResNet with 256 filters and 4*4 kernel, done in PyTorch, for a total of 1.2 M parameters.

- The program does not include a pool of players as in [1], but only one player.

- Evolution of ELO rating during training. ELO's are computed against a pure MCTS player whose ELO ratings have been precomputed (see pre_compute_elo_ratings). We defined the origin by setting the ELO rating of the random player to 0

- A pretrained model that is almost perfect

- You can play against the NN by runing play_against_human.py. You can choose who starts. It's a good way to check optimality of AI moves by comparing to a perfect solver, eg. http://connect4.gamesolver.org/?pos=

## Default settings:

- 40 CPUs

- 400 games played every iteration, with an increasing number of simulations in the MCTS (start : 30 sims, +=1 at every iteration)

- We don't save the previous data of self play (but it can be done, see use_prev_data in config file)

- Training will automatically stop if it reaches an ELO score of 1800 and with good enough probabilities for the first moves (this can be easily modified)

- Other options and parameters are described in the config file

## Particular add-on :
The fact that the first player can always win makes the model oscillates. After the first turn, when yellow player played in the middle, the red player has no incentive to make the best move (in the middle), because he will lose anyway. To counter this, I have included an decreasing reward (z in DeepMind's paper) for the winning player, making him willing to win as fast as possible, and an increasing (negative) reward for the losing player, making him willing to lose, ok, but in the most heroic way (this favors long games). See favor_long_games parameter, set to 0.1 for this run, in config file. It seems to stabilise the NN.

## Getting Started
Run main.py. It will use the pretrained model best_model_resnet.pth. If you want to start from scratch, delete the file in your local copy. First edit the config.py file to set up the number of CPU, and to allow GPU training or not. You can choose between a dense layer network with default two hiden layers of size 1024, for about 2M parameters (still performing poorly), or a ResNet with as many residual block as you want (default is one). Requires torch, in particular.

### Outputs:
At each succesful improvement of the player, the program will in particular print the NN outcomes for 6 particularly important positions, namely when both players first fill the central column (which is the best play).

For instance it will say :

- Turn 0 : Q-value of this board 0.24 (explained : turn 0 is the empty board. The Q-value of the NN, if positive, says that the current player will most likely win the game.)

- Then it says : children probs [0.0, 0.0, 0.0, 99.9, 0.0, 0.0, 0.0]. These are the prior probabilities to chose a child. Here it is a trained network, so it has converged to the best move.

- Finally it gives the Q-value of the possible childs. Here it was : [0.07, -0.09, -0.19, -0.3, 0.02, 0.01, -0.26], meaning that playing in the middle is the best move since this makes the next player to have the lower Q-value of the next board.
MCTS sims are thus run by maximising the PUCT where the leaf value is minus the Q-value.

## Results

The model usually plays almost perfectly after ~150 iterations (several hours with 40 CPUs and 1 GeForce GTX 1080)

Here is the learning curve for the pretrained model given in this repo. Horizontal black lines are the estimated ELO ratings of a pure MCTS with corresponding simulation number.
The red line is the Elo score of the NN player, and was computed at each iteration by playing 80 games against a 800-sims MCTS. In fact, the ELO is underestimated, since you can run elorating.py to show that the pretrained NN actually wins 95% of games against MCTS with 10000 sims (thus ELO more likely of order 2200), and 82.5% against 50000 sims.
Increasing the number of MCTS does not help to defeat the NN. Most likely, the MCTS makes mistakes at the beginning of the game even with many sims, while the NN doesn't (or almost never).

[![Elo ratings](https://github.com/jpbruneton/Alpha-Zero-algorithm-for-Connect-4-game/blob/master/NN_elo_ratings.png)](https://github.com/jpbruneton/Alpha-Zero-algorithm-for-Connect-4-game/blob/master/NN_elo_ratings.png)


## References:

[1] DeepMind AlphaZero : https://www.nature.com/articles/nature24270

[2] Bitboard encoding : http://blog.gamesolver.org/

Other similar projects on Github, in particular:

https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning

https://github.com/blanyal/alpha-zero
