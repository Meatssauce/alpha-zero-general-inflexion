import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .InflexionNNet import InflexionNNet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 512,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        super().__init__(game)
        self.model = InflexionNNet(game)
        self.depth, self.board_x, self.board_y = game.to_planes().shape
        self.action_size = game.max_actions

        if args.cuda:
            self.model.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-4)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.model.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board)
        board = board.unsqueeze(0).float()
        if args.cuda:
            board = board.contiguous().cuda()

        self.model.eval()
        with torch.no_grad():
            pi, v = self.model(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
