import os
import logging
import sys
from multiprocessing import Pool
from pickle import Pickler, Unpickler

import numpy as np
from tqdm import tqdm

from Game import Game
from MCTS import MCTS
from flags import GameStatus
from inflexion.pytorch.NNet import NNetWrapper

log = logging.getLogger(__name__)


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
    trainExamples = []
    episodeStep = 0

    while True:
        episodeStep += 1
        temp = int(episodeStep < args.tempThreshold)

        # get action probabilities from the perspective of current player
        pi = mcts.getActionProb(game, temp=temp)
        sym = game.getSymmetries(game.canonicalBoard, pi)
        for b, p in sym:
            trainExamples.append([b, p, game.player])

        action = np.random.choice(len(pi), p=pi)
        game, curPlayer = game.getNextState(action)

        game.player = curPlayer
        result = game.getGameEnded()

        if result != GameStatus.ONGOING:
            return [(board, policy, result.value if player == curPlayer else -result.value)
                    for board, policy, player in trainExamples]


def main():
    with open('game', "rb") as f, open('args', 'rb') as g:
        game = Unpickler(f).load()
        args = Unpickler(g).load()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint('./temp/', 'mt.pth.bar')

    iterationTrainExamples = []

    with Pool() as p, tqdm(total=args.numEps, desc="Self Play") as pbar:
        items = ((game.reset(), MCTS(nnet, args), args) for _ in range(args.numEps))
        for results in p.imap_unordered(executeEpisode, items):
            iterationTrainExamples += results
            pbar.update()

    with open('iterationTrainExamples', "wb") as f:
        Pickler(f).dump(iterationTrainExamples)


if __name__ == '__main__':
    main()
