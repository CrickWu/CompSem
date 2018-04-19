#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
USAGE: Solve for |F X U - Y|, X is the tgt (dim n), Y is src (dim m)
OUTPUT:
"""

import utils.data_helper
import utils.eval_helper

# insert this to the top of your scripts (usually main.py)
import sys, warnings, traceback, torch, os
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

import time
from tqdm import tqdm
import logging
import argparse

import torch
import torch.nn as nn
from torch.optim import *
from torch.nn import Parameter
from torch.nn import ParameterList
from torch.autograd import Variable
from torch.autograd import grad
import numpy as np
from scipy.stats import ortho_group, zscore
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_small_data(n=4, m=3, dim=2):
    # small test
    # n >= m

    X = torch.randn((n, dim))
    X = X / X.norm(p=2, dim=1, keepdim=True)

    Y_words = map(str, range(m))
    X_words = map(str, range(n))

    Y = X[:m, :]

    # return X, Y
    return Y, Y_words, X, X_words

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch FU Solver')
    parser.add_argument('--data', type=str, required=True, default='../data/', help='location of the data files')
    parser.add_argument('--src', type=str, default='es', help='source language')
    parser.add_argument('--tgt', type=str, default='en', help='target language')

    parser.add_argument('--rand_seed', type=int, default=54321, help='random seed')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--save', type=str, default='save', help='location of saved files')

    # parser.add_argument('--F_file', type=str, default=None, help='whether load F from a (seed) dictionary file')
    # parser.add_argument('--U_file', type=str, default=None, help='whether load U from an existing file')
    parser.add_argument('--F_training', type=str, default=None, help='whether load a F_trn as initialization')
    parser.add_argument('--F_validation', type=str, default=None, help='whether load a F_val to validate')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--log', type=str, default='', help='extra log info')
    parser.add_argument('--log_inter', type=int, default=50, help='log interval')
    parser.add_argument('--normalize', action='store_true', help='whether normalize word embeddings')
    parser.add_argument('--train_size', type=int, default=0, help='the size of seed dictionary')

    args = parser.parse_args()
    return args

def create_logger(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger = logging.getLogger('iter_solver')
    if args.data == 'toy':
        args.log_dir = os.path.join(args.save, 'toy_FU/')
    else:
        args.log_dir = os.path.join(args.save, '{}-{}_FU/'.format(args.src, args.tgt))
    # if not os.path.exists(log_dir):
    mkdir_p(args.log_dir)
    logger.addHandler(logging.StreamHandler())
    # logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, 'info.log')))
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, 'info_{}epoch{}.log'.format(args.log, args.epochs))))
    logger.info(args)
    return logger

def main(args, logger):

    args.cuda = args.gpu is not None
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --gpu")
        else:
            torch.cuda.manual_seed(args.rand_seed)

    if args.data == 'toy':
        m = 3
        n = 3
        dim = 2
        Y, Y_words, X, X_words = get_small_data(n=n, m=m, dim=dim)
        F_data = torch.zeros(m, n)
        F_val = np.zeros((m, n))
        F_val[:m, :m] = np.diag([1] * m)
        U = torch.from_numpy(ortho_group.rvs(dim)).float()
        args.F_validation = 'toy'
    else:
        Y, Y_words, X, X_words = utils.data_helper.get_data(args.data, args.src, args.tgt)
        if args.normalize:
            X = X / np.linalg.norm(X, ord=2, axis=1, keepdims=True)
            Y = Y / np.linalg.norm(Y, ord=2, axis=1, keepdims=True)
        if args.F_validation is not None:
            F_val = utils.data_helper.get_dictionary_matrix(args.F_validation, Y_words, X_words)

        if args.train_size > 0:
            F_trn = utils.data_helper.get_dictionary_matrix(args.F_training, Y_words, X_words, limit=args.train_size)
            nonzero_row, nonzero_col = F_trn.nonzero()
            X_red = torch.from_numpy(X[nonzero_col,:]).float()
            Y_red = torch.from_numpy(Y[nonzero_row,:]).float()
            F_data = torch.from_numpy(F_trn[nonzero_row,:][:,nonzero_col]).float()
            train_set = set(nonzero_row)

            # TODO: should be one-hot per-row
            F_data = F_data / F_data.norm(p=1, dim=1, keepdim=True) # row-sum == 1
            prod_U, prod_sigma, prod_V = torch.svd(Y_red.t().matmul(F_data.matmul(X_red)))
            U = prod_V.matmul(prod_U.t())
        else:
            train_set = set()
        X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    # TODO: sparsity
    H = X.matmul(X.t()) # n x n
    G = Y.matmul(Y.t()) # m x m

    H_s = X.t().matmul(X)
    G_s = Y.t().matmul(Y)


    n = X.size(0)
    m = Y.size(0)
    dim = Y.size(1)
    logger.info('n(tgt) {} m(src) {} dim {}'.format(n, m, dim))

    range_n = torch.LongTensor(range(0, m))
    if args.cuda:
        F_data = F_data.cuda()
        X = X.cuda()
        Y = Y.cuda()
        U = U.cuda()
        H = H.cuda()
        G = G.cuda()
        H_s = H_s.cuda()
        G_s = G_s.cuda()
        range_n = range_n.cuda()

    # forward
    t = [0., 0.]

    try:
        for epoch in tqdm(range(1, args.epochs + 1)):

            start = time.time()

            # Update for F_data
            F_data = F_data.new(m, n).zero_() # m x n scoring matrix of same type
            scores = Y.matmul(X.matmul(U).t()) # m x n
            sorted_scores, indices = torch.max(scores, dim=1, keepdim=False) # 1 x m
            # indices = indices.cpu()
            # print type(range_n), type(indices)
            F_data[range_n, indices] = 1.
            start, elapsed = time.time(), (time.time() - start)
            t[0] += elapsed

            # Update for U
            prod_U, prod_sigma, prod_V = torch.svd(Y.t().matmul(F_data.matmul(X)))
            start, elapsed = time.time(), (time.time() - start)
            U = prod_V.matmul(prod_U.t())
            # print('U U^T',U.matmul(U.t()))
            t[1] += elapsed

            res = F_data.matmul(X).matmul(U) - Y
            loss = torch.sum(res*res) / m

            res1 = F_data.matmul(H).matmul(F_data.t()) - G
            loss1 = torch.sum(res1*res1) / m / m

            res2 = H.matmul(F_data.t()) - F_data.t().matmul(G)
            loss2 = torch.sum(res2*res2) / n / m

            res3 = H_s.matmul(U) - U.matmul(G_s)
            loss3 = torch.sum(res3*res3) / dim / dim

            logger.info('epoch {:3d} | loss {:5.8f} | loss1 {:5.8f} | loss2 {:5.8f} | loss3 {:5.8f}'.format(epoch, loss, loss1, loss2, loss3))

            # check bilingual dictionary induction accuracy(if args.F_validation)
            if args.F_validation is not None:
                logger.info('validation accuracy {:5.8f}'.format(utils.eval_helper.knn_accuracy_from_matrix(F_data.cpu().numpy(), F_val, k=1, src_words=Y_words, tgt_words=X_words, verbose=False)))
    except KeyboardInterrupt:
        logger.info('-' * 10)
        logger.info('Exiting early')

    logger.info('F_solver {:.6} U_solver {:.6}'.format(*t))
    if args.data == 'toy':
        print('F', F_data.cpu())
        print('U', U.cpu())
        print('X', X.cpu())
        print('Y', Y.cpu())
        print('scores', scores.cpu())
    else:
        np.savetxt(os.path.join(args.log_dir, 'U.txt'), U.cpu().numpy())
        utils.data_helper.save_emb(os.path.join(args.log_dir, '{}.iter_solver.vec'.format(args.tgt)), X.matmul(U).cpu().numpy(), X_words)
        utils.data_helper.save_emb(os.path.join(args.log_dir, '{}.iter_solver.vec'.format(args.src)), Y.cpu().numpy(), Y_words)
        # plot_emb(X.matmul(U).cpu().numpy(),Y.cpu().numpy(),'after_orth_transfer.pdf')
        # np.savetxt('{}/F_loss{:.6f}.txt'.format(log_dir, loss), F_data.cpu().numpy(), fmt='%.5e')
        # np.savetxt('{}/U_loss{:.6f}.txt'.format(log_dir

if __name__ == '__main__':
    args = parse_args()
    logger = create_logger(args)
    main(args, logger)