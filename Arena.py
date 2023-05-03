import logging
from collections import Counter
from itertools import cycle
from multiprocessing import Pool

from tqdm import tqdm

from flags import PlayerColour, GameStatus
from inflexion.InflexionPlayers import Player

log = logging.getLogger(__name__)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def playGame(self, player1: Player, player2: Player, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        assert isinstance(player1, Player) and isinstance(player2, Player)

        player = cycle([player1, player2])
        game = self.game.clone()
        assert game.currTurn == 0
        it = 0
        while game.getGameEnded() == GameStatus.ONGOING:
            it += 1
            if verbose:
                print("Turn ", str(it), "Player ", str(game.player.name))
                game.display()
            action = next(player).play(game)

            valids = game.getValidMoves()

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            game, curPlayer = game.getNextState(action)
            game.player = curPlayer

        # Results from red's perspective
        game.player = game.firstMover
        if verbose:
            print("Game over: Turn ", str(it), "Result ", str(game.getGameEnded().name))
            game.display()
        return game.getGameEnded()

    def playGames(self, num: int, verbose=False):
        """
        Plays num games in which player1 starts num//2 games and player2 starts
        num//2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        assert isinstance(num, int) and num >= 2

        wins = {self.player1: 0, self.player2: 0}
        draws = 0

        for i, (player1, player2) in enumerate([(self.player1, self.player2), (self.player2, self.player1)]):
            for _ in tqdm(range(num // 2), desc=f"Arena.playGames ({i})"):
                gameResult = self.playGame(player1=player1, player2=player2, verbose=verbose)
                match gameResult:
                    case GameStatus.DRAW:
                        draws += 1
                    case GameStatus.WON:
                        wins[player1] += 1
                    case GameStatus.LOST:
                        wins[player2] += 1
                    case _:
                        raise ValueError(f'Unexpected game status: {gameResult}')

        return wins[self.player1], wins[self.player2], draws

    #     totalWins = Counter()
    #     totalDraws = 0
    #     total = num // 2
    #
    #     with Pool() as p:
    #         for i, (player1, player2) in enumerate([(self.player1, self.player2), (self.player2, self.player1)]):
    #             args = ((player1, player2, verbose) for _ in range(total))
    #
    #             with tqdm(total=total, desc=f"Arena.playGames ({i})") as pbar:
    #                 for wins, draws in p.imap_unordered(self.getGameResults, args):
    #                     totalWins += wins
    #                     totalDraws += draws
    #                     pbar.update()
    #
    #     return totalWins[hash(self.player1)], totalWins[hash(self.player2)], totalDraws
    #
    # def getGameResults(self, args):
    #     player1, player2, verbose = args
    #     wins = Counter()
    #     draws = 0
    #     gameResult = self.playGame(player1=player1, player2=player2, verbose=verbose)
    #     match gameResult:
    #         case GameStatus.DRAW:
    #             draws += 1
    #         case GameStatus.WON:
    #             wins[hash(player1)] += 1
    #         case GameStatus.LOST:
    #             wins[hash(player2)] += 1
    #         case _:
    #             raise ValueError(f'Unexpected game status: {gameResult}')
    #     return wins, draws
