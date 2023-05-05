import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from torch.multiprocessing import Pool
import torch
from time import sleep

import numpy as np
from tqdm import tqdm

from Arena import Arena
from Game import Game
from MCTS import MCTS
from flags import PlayerColour, GameStatus
from inflexion.InflexionPlayers import MCTSPlayer, RandomPlayer, GreedyPlayer
from inflexion.pytorch.NNet import NNetWrapper

log = logging.getLogger(__name__)
torch.multiprocessing.set_start_method('spawn', force=True)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game: Game, nnet: NNetWrapper, args):
        assert isinstance(game, Game)
        assert isinstance(nnet, NNetWrapper)
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self, args):
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
        game, mcts = args
        assert isinstance(game, Game) and isinstance(mcts, MCTS)
        policyPlanes = []
        boardStacks = []
        players = []
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            # Get action probabilities from the perspective of current player
            # temp1 = game.board.copy()
            pi = mcts.getActionProb(game, temp=temp)
            assert isinstance(pi, np.ndarray) and pi.size == game.getActionSize()

            policyPlane = pi.reshape(game.policyShape)
            boardStack = game.toNNetInput()

            policyPlanes += game.symmetries(policyPlane)
            boardStacks += game.symmetries(boardStack)
            players += [game.player] * len(policyPlanes)

            # assert (game.board == temp1).all()
            action = np.random.choice(len(pi), p=pi)
            game, curPlayer = game.getNextState(action)

            result = game.getGameEnded()

            if result == GameStatus.ONGOING:
                continue

            return [(board, policy.ravel().tolist(), result.value if player == curPlayer else -result.value)
                    for board, policy, player in zip(boardStacks, policyPlanes, players)]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        pitInterval = 5
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    mcts = MCTS(self.nnet, self.args)  # reset search tree
                    game = self.game.reset()  # reset game
                    iterationTrainExamples += self.executeEpisode((game, mcts))

                # os.makedirs(self.args.sharedPath, exist_ok=True)
                # with open(os.path.join(self.args.sharedPath, 'game'), "wb") as f, \
                #         open(os.path.join(self.args.sharedPath, 'args'), 'wb') as g:
                #     Pickler(f).dump(self.game)
                #     Pickler(g).dump(self.args)
                # self.nnet.save_checkpoint(self.args.sharedPath, 'nnet.pth.bar')
                # os.system("python selfplay.py")
                # while not os.path.exists(os.path.join(self.args.sharedPath, 'iterationTrainExamples')):
                #     sleep(1)
                # with open(os.path.join(self.args.sharedPath, 'iterationTrainExamples'), "rb") as f:
                #     iterationTrainExamples += Unpickler(f).load()
                
                # with Pool() as p, tqdm(total=self.args.numEps, desc="Self Play") as pbar:
                #     items = ((self.game, MCTS(self.nnet, self.args)) for _ in range(self.args.numEps))
                #     for results in p.imap_unordered(self.executeEpisode, items):
                #         iterationTrainExamples += results
                #         pbar.update()
                #     sleep(3)

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = "
                    f"{len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network
            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            if i % pitInterval != 0:
                continue

            log.info('PITTING AGAINST BASELINES')
            nmcts = MCTS(self.nnet, self.args)
            player1 = MCTSPlayer(nmcts)

            player2 = RandomPlayer()
            arena = Arena(player1, player2, self.game.reset())
            p1wins, p2wins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/RANDOM WINS : %d / %d ; DRAWS : %d' % (p1wins, p2wins, draws))

            player2 = GreedyPlayer()
            arena = Arena(player1, player2, self.game.reset())
            p1wins, p2wins, draws = arena.playGames(self.args.arenaCompare)
            log.info('NEW/GREEDY WINS : %d / %d ; DRAWS : %d' % (p1wins, p2wins, draws))

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

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
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
