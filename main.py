import logging

import coloredlogs

from Coach import Coach
from inflexion.InflexionGame import InflexionGame as Game
from inflexion.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(7, max_turns=343, max_power=6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    
    load_folder_file = ('./dev/models/inflexion/7x343x6', 'best5.pth.tar')
    load_model = False

    log.info('Loading the Coach...')
    c = Coach(g, nnet, load_folder_file=load_folder_file)
    
    if load_model:
        log.info('Loading checkpoint "%s/%s"...', *load_folder_file)
        nnet.load_checkpoint(*load_folder_file)
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()