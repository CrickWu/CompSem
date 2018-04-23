#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
USAGE: Solve for |F X U - Y|, X is the tgt (dim n), Y is src (dim m)
OUTPUT:
"""

import utils.data_helper
import utils.eval_helper
import utils.word2gm_loader

# insert this to the top of your scripts (usually main.py)
#import sys, warnings, traceback, torch, os
#def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#    traceback.print_stack(sys._getframe(2))
#warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
#torch.utils.backcompat.broadcast_warning.enabled = True
#torch.utils.backcompat.keepdim_warning.enabled = True

import os
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
    parser.add_argument('--src_word2gm_modefile', type=str, required=True, default=None, help='location of src word2gm model directiory')
    parser.add_argument('--tgt_word2gm_modefile', type=str, required=True, default=None, help='location of tgt word2gm model directiory')
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

def load_w2gm_model(w2gm_model):
    assert(os.path.isdir(w2gm_model))
    w2gm_model = utils.word2gm_loader.Word2GM(save_path=w2gm_model)
    #mus:    N x K x D
    #sigs:   N x K x 1
    #alphas: N x K
    def model_to_dict(model):
        m_dict = {'mus': model.mus,
                  'sigs': np.exp(model.logsigs),
                  'alphas': model.mixture,
                  'id2word': model.id2word}
        return m_dict

    return model_to_dict(w2gm_model)

# expected likelihood kernel
# between f^s_i and f^t_j
# X_mus: N x K x D
# X_sigs: N x K x 1
# X_alphas: N x K
def dist_dot(X_mus, X_sigs, X_alphas,
             Y_mus, Y_sigs, Y_alphas,
             V, w1, w2):
    mu1, mu2 = X_mus[w1], Y_mus[w2]
    sigma1, sigma2 = X_sigs[w1], Y_sigs[w2]
    mix1, mix2 = X_alphas[w1], Y_alphas[w2]
    num_mix = X_mus.size(1)

    # rotate mu1 by V
    mu1 = torch.matmul(mu1, V)
    def partial_energy(cl1, cl2):
        # cl1, cl2 are 'cluster' indices
        _a = sigma1[cl1] + sigma2[cl2]
        _res = -0.5*torch.sum(torch.log(_a)) #TODO: shouldn't this times D? 0.5 * D * log(sigma1+ sigma2)
        ss_inv = 1./_a
        diff = mu1[cl1] - mu2[cl2]
        _res += -0.5*torch.sum(diff*ss_inv*diff)
        return _res

    partial_energies = torch.zeros((num_mix, num_mix))
    partial_energies = Variable(partial_energies.cuda())
    for _i in range(num_mix):
        for _j in range(num_mix):
            partial_energies[_i,_j] = partial_energy(_i, _j)

    # for numerical stability
    max_partial_energy = torch.max(partial_energies)
    energy = 0
    for _i in range(num_mix):
        for _j in range(num_mix):
            energy += mix1[_i]*mix2[_j]*torch.exp(partial_energies[_i,_j] - max_partial_energy)
    log_energy = max_partial_energy + torch.log(energy)
    #print('log_energy', type(log_energy), log_energy.size())
    return log_energy

def update_U(X_mus, X_sigs, X_alphas,
             Y_mus, Y_sigs, Y_alphas,
             U, V, range_n):

    N, K, D = X_mus.size()
    M = Y_mus.size(0)
    U_new = U.new(M, N).zero_()   # m x n scoring matrix of same type
    scores = U.new(M, N).zero_() #
    for  m in range(M):
        for n in range(N):
            tmp = dist_dot(X_mus, X_sigs, X_alphas, Y_mus, Y_sigs, Y_alphas, V, n, m).data.cpu()
            scores[m, n] = tmp[0]
            #scores[m, n] = dist_dot(X_mus, X_sigs, X_alphas,
            #                        Y_mus, Y_sigs, Y_alphas,
            #                        V, n, m).data.cpu()
            #exit(0)
    sorted_scores, indices = torch.max(scores, dim=1, keepdim=False) # 1 x m
    # indices = indices.cpu()
    # print type(range_n), type(indices)
    U_new[range_n, indices] = 1.
    return U_new

def update_V(U, V, X, Y):
    return None

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

    # load word2gm model
    X_dict = load_w2gm_model(args.src_word2gm_modefile)
    Y_dict = load_w2gm_model(args.tgt_word2gm_modefile)

    # mus: N x K x D
    # sigs: N x K x 1 (spherical Guassian)
    # alphas: N x K
    X_mus = torch.from_numpy(X_dict['mus']).float()
    Y_mus = torch.from_numpy(Y_dict['mus']).float()
    X_sigs = torch.from_numpy(X_dict['sigs']).float()
    Y_sigs = torch.from_numpy(Y_dict['sigs']).float()
    X_alphas = torch.from_numpy(X_dict['alphas']).float()
    Y_alphas = torch.from_numpy(Y_dict['alphas']).float()

    #CONSTANT
    N, K, D = X_mus.size()
    M = Y_mus.size(0)
    assert(K == Y_mus.size(1))
    assert(D == Y_mus.size(2))
    logger.info('N(src) {} M(tgt) {} D {} K {}'.format(N, M, D, K))

    # Parameter to be optimized
    # U: M x N rowise one-hot matrix
    # V: D x D orthogonal matrix
    # WARNING: U should not be put in GPU!!
    U = torch.zeros(M, N)
    V = torch.from_numpy(ortho_group.rvs(D)).float()
    range_n = torch.LongTensor(range(0, M))

    if args.cuda:
        X_mus = Variable(X_mus.cuda())
        Y_mus = Variable(Y_mus.cuda())
        X_sigs = Variable(X_sigs.cuda())
        Y_sigs = Variable(Y_sigs.cuda())
        X_alphas = Variable(X_alphas.cuda())
        Y_alphas = Variable(Y_alphas.cuda())
        # range_n = range_n.cuda()
        #U = Variable(U.cuda())
        V = Variable(V.cuda())

    # forward
    t = [0., 0.]
    try:
        for epoch in tqdm(range(1, args.epochs + 1)):

            start = time.time()

            # Update for U
            U_new = update_U(X_mus, X_sigs, X_alphas,
                             Y_mus, Y_sigs, Y_alphas,
                             U, V, range_n)
            start, elapsed = time.time(), (time.time() - start)
            t[0] += elapsed
            exit(0)

            # Update for V
            V_new = update_V(X_mus, X_sigs, X_alphas,
                             Y_mus, Y_sigs, Y_alphas,
                             U, V)
            start, elapsed = time.time(), (time.time() - start)
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
