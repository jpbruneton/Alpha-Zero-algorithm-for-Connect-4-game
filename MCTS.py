#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             MCTS.py
# Description:      Pure MCTS with random rollout policy
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
from Game_bitboard import Game
import random
# ============================================================================ #


# =============================== CLASS: NODE ================================ #
# A class representing a node of a mcts
class Node:
    # ---------------------------------------------------------------------------- #
    # Constructs a node from a state
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move #is the move (an int) that was played from parent to get there
        self.parent = parent
        self.children = []

        self.N = 0  # vists
        self.W = 0  # cumulative reward
        self.Q = 0  # average reward

    def isLeaf(self):
        return len(self.children) == 0

    def isterminal(self):
        game = Game(self.state)
        gameover, _ = game.gameover() #is 0 or 1
        return gameover

# ============================================================================ #


# =============================== CLASS: MCTS ================================ #
# A class representing a mcts
class MCTS:
    # ---------------------------------------------------------------------------- #
    # Constructs a tree
    def __init__(self):
        self.root = None

    # ---------------------------------------------------------------------------- #
    # Builds a node from the state and adds it to the tree
    def createNode(self, state, move=None, parent=None):
        node = Node(state, move, parent)
        return node

    # ---------------------------------------------------------------------------- #
    # Picks a leaf according to UCT formula
    def selection(self, node, c_uct, evaluator):

        # if the provided node is already a leaf (it shall happen only at the first sim)
        if node.isLeaf():
            return node, node.isterminal()

        else:  # the given node is not a leaf, thus pick a leaf descending from the node, according to UCT :
            current = node
            while not current.isLeaf():
                values = np.empty(len(current.children))
                values[:] = np.asarray([evaluator(node, c_uct) for node in current.children])
                posmax = np.where(values == np.max(values))[0]
                imax= posmax[int(random.random() * len(posmax))]
                # Moves the current to the next
                current = current.children[imax]

            # it is entirely possible that the chosen leaf is terminal:
        return current, current.isterminal()

    # ---------------------------------------------------------------------------- #
    def expand_all(self, node):
        game=Game(node.state)
        allowed_moves = game.allowed_moves()
        for move in allowed_moves:
            child = self.createNode(game.nextstate(move), move, parent=node)
            node.children += [child]

    # ---------------------------------------------------------------------------- #
    # Makes a rollout starting by a particular node
    def default_rollout_policy(self, node, usecounter):

        gameloc=Game(node.state)

        if node.isterminal() == 0:
            #init
            allowedmoves = gameloc.allowed_moves()
            gameover = 0

            # completely random rollout/ or random rollout but take the win or counter the lose, is usecounter
            while gameover == 0:

                if usecounter:
                    can_win, where_win, can_lose, where_lose = gameloc.iscritical()
                    if can_win:
                        move = where_win[int(random.random() * len(where_win))]
                        gameloc.takestep(move)
                        allowedmoves = gameloc.allowed_moves()
                        gameover,_ = gameloc.gameover()

                    elif can_lose:
                        imax = where_lose[int(random.random() * len(where_lose))]
                        gameloc.takestep(imax)
                        allowedmoves = gameloc.allowed_moves()
                        gameover,_ = gameloc.gameover()

                    else:
                        randommove = allowedmoves[int(random.random() * len(allowedmoves))]
                        gameloc.takestep(randommove)
                        allowedmoves = gameloc.allowed_moves()
                        gameover,_ = gameloc.gameover()
                else:
                    randommove = allowedmoves[int(random.random() * len(allowedmoves))]
                    gameloc.takestep(randommove)
                    allowedmoves = gameloc.allowed_moves()
                    gameover, _ = gameloc.gameover()

        _, winner = gameloc.gameover()

        return winner

    # ---------------------------------------------------------------------------- #
    # Back propagates values after simulations
    def regular_back_prop(self, child, result, whoplayatleaf):

        if result == 0:
            newreward=0
            child.N += 1
            child.Q = child.W / child.N
        else:
            newreward = result * whoplayatleaf
            child.W += newreward
            child.N += 1
            child.Q = child.W / child.N

        # Then init recursion
        current = child
        count=1
        while current.parent is not None:
            current.parent.N += 1
            current.parent.W += ((-1)**count)*newreward
            current.parent.Q = current.parent.W / current.parent.N
            # move up
            current = current.parent
            count += 1

    # -------------------------------------------------------------ee--------------- #
    # Back propagates values after simulations
    def back_prop_terminal(self, leaf_terminal):

        game = Game(leaf_terminal.state)
        gameover, winner = game.gameover()

        if winner == 0:
            leaf_terminal.W += 0
            leaf_terminal.N += 1
            leaf_terminal.Q = leaf_terminal.W / leaf_terminal.N
            new_reward=0

        else: #a terminal leaf is always a draw or reward 1 (for the player that played the move)
            new_reward = 1
            leaf_terminal.W += new_reward
            leaf_terminal.N += 1
            leaf_terminal.Q = leaf_terminal.W / leaf_terminal.N

        # Then init recursion
        current = leaf_terminal
        count=1

        while current.parent is not None:
            current.parent.N += 1
            current.parent.W += ((-1)**count)*new_reward
            current.parent.Q = current.parent.W / current.parent.N
            # move up
            current = current.parent
            count+=1

    # ---------------------------------------------------------------------------- #
    # Runs a entire simulation in the tree
    def simulate(self, node, evaluator, c_uct, usecounter):

        leaf, isleafterminal = self.selection(node, c_uct, evaluator)

        if isleafterminal == 0:
            #expansion
            whoplay_at_leaf = int(leaf.state[2])
            self.expand_all(leaf)

            #rollout only once, and only one of the child
            index = random.randint(0, len(leaf.children) - 1)
            child = leaf.children[index]
            result = self.default_rollout_policy(child, usecounter)

            #update newly added child and its ancestors
            self.regular_back_prop(child, result, whoplay_at_leaf)

        else:
            self.back_prop_terminal(leaf)

# ============================================================================ #