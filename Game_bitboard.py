#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             Game_bitboard.py
# Description:      Game environment using a 64 bit encoding of the board
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import config
import numpy as np
# ============================================================================

# Important Note. This file uses a bitboard encoding. A state for yellow or red is a 64 bit integer,
# with the following bitboard encoding
#
# 5  13 21  29  37  45  53
# 4  12 20  28  36  44  52
# 3  11 19  27  35  43  51
# 2  10 18  26  34  42  50
# 1  9  17  25  33  41  49
# 0  8  16  24  32  40  48
#
# This is way faster (by a factor ~ 45) than using arrays or lists.
# Drawback is that the following code is very specific to a 6*7 board
# Choice of encoding taken from : http://blog.gamesolver.org/

# =============================== CLASS: Game ================================ #

class Game:
# ---------------------------------------------------------------------------- #
# Constructor

    def __init__(self, state=None):

        self.H=6
        self.L=7
        # a state is three numbers : an integer for yellow, one for red, and player_turn

        if state is None:
            #initiate a new game, by default : yellow (playerturn =1) always starts
            self.yellowstate = 0
            self.redstate = 0
            self.player_turn = 1
            self.state=[self.yellowstate,self.redstate,self.player_turn]

        else:
            self.state = state
            self.yellowstate = self.state[0]
            self.redstate = self.state[1]
            self.player_turn = self.state[2]

# ---------------------------------------------------------------------------- #
# check if there is a win for a given board's player.
    def checkwin(self, onecolorboard):
        # check horizontal
        horizontal = onecolorboard & (onecolorboard >> 8)
        horizontal &= horizontal >> 16

        # check vertical
        vertical = onecolorboard & (onecolorboard >> 1)
        vertical &= vertical >> 2

        # check diagonal /
        diagonal1 = onecolorboard & (onecolorboard >> 9)
        diagonal1 &= diagonal1 >> 18

        # check diagonal \
        diagonal2 = onecolorboard & (onecolorboard >> 7)
        diagonal2 &= diagonal2 << 14

        win = horizontal | vertical | diagonal1 | diagonal2

        return win

# ---------------------------------------------------------------------------- #
# useful function, see below
    def bitcounter(self, v):
        r = 0
        while v > 1:
            v >>= 1
            r += 1
        return r

# ---------------------------------------------------------------------------- #
# returns a list of allowed moves. A move here is the integer corresponding to where you can play
    def allowed_moves(self):
        fullboard = self.yellowstate | self.redstate
        allowed_moves=[]

        #try and add a piece to each column
        for i in range(7):
            shift = fullboard >> 8 * i
            addone = shift + 1
            mask = 2 ** 7 - 1
            # append to allowed move only if it is legal (column is not full)
            r = self.bitcounter(mask & addone)
            if r < 6:
                unshift = addone << 8 * i
                allowed_moves.append((unshift | fullboard) ^ fullboard)

        return allowed_moves

# ---------------------------------------------------------------------------- #
# if we want to enforce to take the win or counter a lose. See MCTS_NN and MCTS

    def iscritical(self):

        can_win = 0
        winningmoves = []
        can_lose = 0
        losingmoves = []

        allowed_moves = self.allowed_moves()

        #make a virtual move and check if it wins
        for move in allowed_moves:
            virtual_state = self.nextstate(move)
            if self.player_turn == 1:
                win = self.checkwin(virtual_state[0])
            else:
                win = self.checkwin(virtual_state[1])

            if win:
                can_win=1
                winningmoves.append(move)

            #now play as if the opponent could play right now:
            virtual_state2 = self.nextstate(move, - self.player_turn)
            if self.player_turn == 1:
                lose = self.checkwin(virtual_state2[1])
            else:
                lose = self.checkwin(virtual_state2[0])

            if lose:
                can_lose=1
                losingmoves.append(move)

        return can_win, winningmoves, can_lose, losingmoves

# ---------------------------------------------------------------------------- #
# check gameover and returns winner
    def gameover(self):

        winner = 0

        if self.checkwin(self.state[0]):
            gameover = 1
            winner = 1
            #yellow wins

        elif self.checkwin(self.state[1]):
            gameover = 1
            winner = -1
            # red wins

        elif str(bin(self.state[0])).count('1') + str(bin(self.state[1])).count('1') == 42:
            gameover = 1
            #full board and a draw (winner = 0)

        else :
            gameover =0

        return gameover, winner

# ---------------------------------------------------------------------------- #
# compute next state given a move and who's playing

    def nextstate(self, move, playerturn=None):

        if playerturn is None:
            if self.player_turn==1:
                return [self.state[0] | move, self.state[1], - self.player_turn]
            else:
                return [self.state[0], self.state[1] | move , - self.player_turn]

        else:
            if playerturn==1:
                return [self.state[0] | move, self.state[1], - playerturn]
            else:
                return [self.state[0], self.state[1] | move , - playerturn]

# ---------------------------------------------------------------------------- #
# make the move
    def takestep(self,move):
        self.state = self.nextstate(move)
        self.yellowstate = self.state[0]
        self.redstate = self.state[1]
        self.player_turn = self.state[2]



# ---------------------------------------------------------------------------- #
    def binarystatetoflatlist(self, state):

    # this transforms the state given as an integer into a list (for future entry to the NN)
    # with the convention
    # 0 1 2 3 4 5 6 7
    # 8 9 10 ....
    # ..
    # ..
    # ..
    # ..    ..     42

        tostring = '0' * (64 - len(str(bin(state))[2:])) + str(bin(state))[2:]
        reverse_string = tostring[::-1]

        flatstr = ''
        line = 5
        while line >= 0:
            for col in range(7):
                flatstr += reverse_string[line + 8 * col]
            line -= 1

        tolist = [int(x) for x in flatstr]

        return tolist


# ---------------------------------------------------------------------------- #
# calls the previous function to return a flattened 3*42 state
# with yellow board, red board, and player turn

    def state_flattener(self,state):
        yellow = np.array(self.binarystatetoflatlist(state[0]),dtype=int)
        red = np.array(self.binarystatetoflatlist(state[1]),dtype=int)
        player = np.ones(42,dtype=int)*int(state[2])
        flat=np.hstack((yellow,red,player))
        return flat

# ---------------------------------------------------------------------------- #
# at some point we need to know that one move given as an integer corresponds to one child in the tree
# we thus need to convert a move to a column index (0, ..., 6) :
    def convert_move_to_col_index(self, move):
        r= self.bitcounter(move)
        return r // 8

# ---------------------------------------------------------------------------- #
#for displaying the board

    def display_it(self):
        yellow = self.binarystatetoflatlist(self.state[0])
        red = self.binarystatetoflatlist(self.state[1])
        board = np.array([x - y for (x,y) in zip(yellow,red)])
        board = board.reshape((6,7))
        for i in range(self.H):
            line = []
            for elem in board[i, :]:
                if elem == 0:
                    line.append(' ')
                elif elem == 1:
                    line.append('o')
                    # yellow token
                else:
                    line.append('x')
                    # red token

            print(line)


# =============================== END CLASS: Game ================================ #
