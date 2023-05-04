import logging
import math

import numpy as np

from Game import Game
from flags import GameStatus
from inflexion.pytorch.NNet import NNetWrapper

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet: NNetWrapper, args):
        assert isinstance(nnet, NNetWrapper)
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, game, temp=1) -> np.ndarray:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a 1d ndarray where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        assert isinstance(game, Game)
        assert isinstance(temp, (int, float)) and temp >= 0

        for i in range(self.args.numMCTSSims):
            self.search(game)

        s = game.getCanonicalBoard().tobytes()
        counts = np.array([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(game.getActionSize())])

        if temp == 0:
            bestAs = np.argwhere(counts == np.max(counts)).ravel()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(len(counts), dtype=np.int8)
            probs[bestA] = 1
            return probs

        counts = counts ** (1. / temp)
        probs = counts / counts.sum()
        return probs

    def search(self, game):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        assert isinstance(game, Game)

        s = game.getCanonicalBoard().tobytes()

        if (gameStatus := game.getGameEnded()) != GameStatus.ONGOING:
            # terminal node
            return -gameStatus.value

        if s not in self.Ps:
            # leaf node
            policies, v = self.nnet.predict(game.getCanonicalBoard())
            valids = game.getValidMoves()
            policies *= valids  # masking invalid moves
            sum_Ps_s = policies.sum()

            if sum_Ps_s > 0:
                policies /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                policies += valids
                policies /= policies.sum()

            self.Ps[s] = policies
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = game.getNextState(a)
        next_s.player = next_player

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
