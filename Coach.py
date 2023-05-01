import logging
import os
import sys
from collections import deque
from itertools import count
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from flags import PlayerColour, GameStatus
from inflexion.InflexionPlayers import MCTSPlayer

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.nnet, self.args)
        self.train_data_history = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
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
        train_data = []

        for episodeStep in count(start=1):
            temperature = 1 if episodeStep < self.args.tempThreshold else 0

            pi = self.mcts.getActionProb(self.game, temp=temperature)
            for board, policy in zip(self.game.getSymmetries(self.game.canonical_board),
                                     self.game.getSymmetries(pi.reshape(self.game.policy_shape))):
                train_data.append((board, policy.ravel(), self.game.player))

            action = np.random.choice(len(pi), p=pi)
            # prev = self.game.board.copy()
            # repr = self.game.actionToMove(action)
            self.game, curPlayer = self.game.getNextState(action)
            result = self.game.game_status.score()

            if self.game.game_status != GameStatus.ONGOING:
                return [(board, pi, result if player == self.game.player else -result)
                        for board, pi, player in train_data]

            self.game.player = curPlayer
        raise RuntimeError("Should not reach here")

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i_generation in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Gen #{i_generation} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i_generation > 1:
                train_data = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.nnet, self.args)  # reset search tree
                    self.game = self.game.reset()
                    train_data += self.executeEpisode()

                # save the iteration examples to the history 
                self.train_data_history.append(train_data)

            if len(self.train_data_history) > self.args.numItersForTrainExamplesHistory:
                log.warning(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = "
                            f"{len(self.train_data_history)}")
                self.train_data_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i_generation - 1)

            # shuffle examples before training
            train_data = []
            for e in self.train_data_history:
                train_data.extend(e)
            shuffle(train_data)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet, self.args)

            self.nnet.train(train_data)
            nmcts = MCTS(self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(MCTSPlayer(pmcts), MCTSPlayer(nmcts), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i_generation))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_data_history)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.train_data_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
