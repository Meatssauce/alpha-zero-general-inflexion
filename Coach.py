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
from flags import PlayerColour, GameOutcome
from inflexion.InflexionPlayers import MCTSPlayer, RandomPlayer, GreedyPlayer
from inflexion.pytorch.NNet import NNetWrapper

log = logging.getLogger(__name__)
torch.multiprocessing.set_start_method('spawn', force=True)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet.
    """

    def __init__(self, game: Game, 
                 nnet: NNetWrapper, 
                 tempThreshold=30, 
                 numIters=1000, 
                 maxlenOfQueue=200000,  # Number of game examples to train the neural networks.
                 numEps=30,  # Number of complete self-play games to simulate during a new iteration.
                 numItersForTrainExamplesHistory=20,
                 numMCTSSims=250,
                 cpuct=3,
                 checkpoint='./temp/', 
                 updateThreshold=0.55,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
                 load_folder_file=('./dev/models/inflexion/7x343x6', 'best5.pth.tar')
                ):
        assert isinstance(game, Game)
        assert isinstance(nnet, NNetWrapper)
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.tempThreshold = tempThreshold
        self.numIters = numIters
        self.maxlenOfQueue = maxlenOfQueue
        self.numEps = numEps
        self.numItersForTrainExamplesHistory = numItersForTrainExamplesHistory
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        self.checkpoint = checkpoint
        self.updateThreshold = updateThreshold
        self.load_folder_file = load_folder_file
        self.trainExamplesHistory = []  # history of examples from numItersForTrainExamplesHistory latest iterations
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
        assert game._curr_turn == 0 and game.outcome == GameOutcome.ONGOING

        policyPlanes = []
        boardStacks = []
        players = []
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.tempThreshold)

            # Get action probabilities from the perspective of current player
            pi = mcts.getActionProb(game, temp=temp)
            assert isinstance(pi, np.ndarray) and pi.size == game.max_actions

            policyPlane = pi.reshape(game.policy_shape)
            boardStack = game.to_planes()

            policyPlanes += game.symmetries(policyPlane)
            boardStacks += game.symmetries(boardStack)
            players += [game.player] * len(policyPlanes)

            action = np.random.choice(len(pi), p=pi)
            game = game.to_next_state(action)

            result = game.outcome

            if result == GameOutcome.ONGOING:
                continue
            
            return [(board, policy.ravel().tolist(), result.value if player == game.player else -result.value)
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
        quick_reload_path = 'temp/checkpoint_0.pth.tar.examples'
        for i in range(1, self.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.maxlenOfQueue)
                
                if quick_reload_path is not None:
                    with open(quick_reload_path, "rb") as f:
                        self.trainExamplesHistory = Unpickler(f).load()
                    quick_reload_path = None
                else:
                    # for _ in tqdm(range(self.numEps), desc="Self Play"):
                    #     mcts = MCTS(self.nnet, self.numMCTSSims, self.cpuct)  # reset search tree
                    #     game = self.game.restarted()  # reset game
                    #     iterationTrainExamples += self.executeEpisode((game, mcts))

                    with Pool() as p, tqdm(total=self.numEps, desc="Self Play") as pbar:
                        items = ((self.game, MCTS(self.nnet, self.numMCTSSims, self.cpuct)) for _ in range(self.numEps))
                        for results in p.imap_unordered(self.executeEpisode, items):
                            iterationTrainExamples += results
                            pbar.update()
                    torch.cuda.empty_cache()

                    # save the iteration examples to the history 
                    self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.numItersForTrainExamplesHistory:
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
            self.nnet.save_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            pmp = MCTSPlayer(MCTS(self.pnet, self.numMCTSSims, self.cpuct))
            
            self.nnet.train(trainExamples, epochs=20, batch_size=256)
            self.nnet.save_checkpoint(folder=self.checkpoint, filename=f"iter{i:05d}.pt")

            log.info('PITTING AGAINST BASELINES')
            nmp = MCTSPlayer(MCTS(self.nnet, self.numMCTSSims, self.cpuct))

            arena = Arena(pmp, nmp, self.game.restarted())
            pwins, nwins, draws = arena.playGames(40)  # Number of games to play during arena play to determine if new net will be accepted.
            log.info(f'NEW/PREV WINS : %d / %d ; DRAWS : %d' % (pwins, pwins, draws))
            
            if pwins + nwins == 0 or nwins / (pwins + nwins) < self.updateThreshold:
                print("rejected new model")
                self.nnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            else:
                print("accepting new model")
                self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        modelFile = os.path.join(self.load_folder_file[0], self.load_folder_file[1])
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
