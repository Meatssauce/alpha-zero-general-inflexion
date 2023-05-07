import os
import logging
import sys
from torch.multiprocessing import Pool
from pickle import Pickler, Unpickler

import numpy as np
from tqdm import tqdm
import torch

from Game import Game
from MCTS import MCTS
from flags import GameOutcome
from inflexion.pytorch.NNet import NNetWrapper

log = logging.getLogger(__name__)
torch.multiprocessing.set_start_method('spawn', force=True)


def executeEpisode(items):
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                       pi is the MCTS informed policy vector, v is +1 if
                       the player eventually won the game, else -1.
    """
    game, mcts, args = items
    assert isinstance(game, Game) and isinstance(mcts, MCTS)
    policyPlanes = []
    boardStacks = []
    players = []
    episodeStep = 0

    while True:
        episodeStep += 1
        temp = int(episodeStep < args.tempThreshold)

        # get action probabilities from the perspective of current player
        pi = mcts.getActionProb(game, temp=temp)
        assert isinstance(pi, np.ndarray) and pi.size == game.max_actions

        policyPlane = pi.reshape(game.policyShape)
        boardStack = game.to_planes()

        policyPlanes += game.symmetries(policyPlane)
        boardStacks += game.symmetries(boardStack)
        players += [game.player] * len(policyPlanes)

        # assert (game.board == temp1).all()
        action = np.random.choice(len(pi), p=pi)
        game = game.to_next_state(action)

        result = game.outcome

        if result == GameOutcome.ONGOING:
            continue

        return [(board, policy.ravel().tolist(), result.value if player == game.player else -result.value)
                for board, policy, player in zip(boardStacks, policyPlanes, players)]


def main():
    with open('./shared/game', "rb") as f, open('./shared/args', 'rb') as g:
        game = Unpickler(f).load()
        args = Unpickler(g).load()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(folder='./shared', filename='nnet.pth.bar')
    mcts = MCTS(nnet, args)

    iterationTrainExamples = []

    with Pool() as p, tqdm(total=args.numEps, desc="Self Play") as pbar:
        items = ((game, mcts.reset(), args) for _ in range(args.numEps))
        for results in p.imap_unordered(executeEpisode, items):
            iterationTrainExamples += results
            pbar.update()

    with open('./shared/iterationTrainExamples', "wb") as f:
        Pickler(f).dump(iterationTrainExamples)


if __name__ == '__main__':
    main()
