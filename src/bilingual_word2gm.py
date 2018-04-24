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
    parser = argparse.ArgumentParser(description='PyTorch UV Solver')
    parser.add_argument('--data', type=str, required=True, default='../data/europarl', help='location of the data files')
    parser.add_argument('--src', type=str, default='es', help='source language')
    parser.add_argument('--tgt', type=str, default='en', help='target language')
    parser.add_argument('--rand_seed', type=int, default=54321, help='random seed')
    parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
    parser.add_argument('--save', type=str, default='save', help='location of saved files')
    parser.add_argument('--emb_model', type=str, required=True, default=None, help='location of embedding directiory')
    parser.add_argument('--F_training', type=str, default=None, help='whether load a F_trn as initialization')
    parser.add_argument('--F_validation', type=str, default=None, help='whether load a F_val to validate')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--log', type=str, default='', help='extra log info')
    parser.add_argument('--log_inter', type=int, default=50, help='log interval')
    parser.add_argument('--normalize', action='store_true', help='whether normalize word embeddings')
    parser.add_argument('--train_size', type=int, default=0, help='the size of seed dictionary')
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_init_size", type=int, default=10, help="initial size of seeding dictionary as supervision")
    parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_bs", type=int, default=100, help="batch size for generating dictionary")
    # mapping parameters (for refinement)
    parser.add_argument("--map_lr", type=float, default=1e-3, help="learning rate for update_V")
    parser.add_argument("--map_iter", type=int, default=5, help="learning rate for update_V")
    args = parser.parse_args()
    return args

def create_logger(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger = logging.getLogger('iter_solver')
    if args.data == 'toy':
        args.log_dir = os.path.join(args.save, 'toy_%s' % (args.emb_model))
    else:
        args.log_dir = os.path.join(args.save, '{}-{}_{}/'.format(args.src, args.tgt, args.emb_model))
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
# between mini-batch of X and all Y
def kronecker_op(X, Y):
    """
    Input:
        X: (a, b, D)
        Y: (m, n, D)
    Output:
        expand_X: (a*m, b*n, D)
        tile_Y:   (a*m, b*n, D)
    """
    a, b, Dx = X.shape
    m, n, Dy = Y.shape
    assert Dx == Dy
    D = Dx
    tile_Y = Y.repeat(a, b, 1)
    expand_X = X.view(a, b, 1, D).repeat(1, m, n, 1).view(a*m, b*n, D)
    return expand_X, tile_Y

# X_mus: bs x K x D
# Y_mus: M x K x D
# X_sigs: bs x K x 1
# Y_sigs: M x K x 1
# X_alphas: bs x K
# Y_alphas: M x K
def distdot_outer(X_mus, X_sigs, X_alphas,
                  Y_mus, Y_sigs, Y_alphas):
    bs, K, D = X_mus.size()
    M = Y_mus.size(0)
    # calcuate (X_mus - Y_mus)^T (X_mus - Y_mus)
    # X_mus_r: bs x M x K x K x D
    X_mus_kop, Y_mus_kop = kronecker_op(X_mus, Y_mus)
    X_minus_Y = (X_mus_kop - Y_mus_kop).view(-1, D)
    XY_dot = torch.sum(X_minus_Y * X_minus_Y, 1).view(bs*M, K*K)

    # calculate X_sigs + Y_sigs
    X_sigs_kop, Y_sigs_kop = kronecker_op(X_sigs, Y_sigs)
    XY_sig = (X_sigs_kop + Y_sigs_kop).view(bs*M, K*K)

    # calculate xi: (bsxM) x (KxK)
    partial_energies = -0.5 * D * torch.log(XY_sig) -0.5 * D * np.log(np.pi)- 0.5 * (1. / XY_sig) * XY_dot
    partial_energies = torch.exp(partial_energies)

    # calculate energy
    X_alphas_kop, Y_alphas_kop = kronecker_op(X_alphas.unsqueeze(2), Y_alphas.unsqueeze(2))
    X_alphas_kop, Y_alphas_kop = X_alphas_kop.squeeze(2), Y_alphas_kop.squeeze(2)
    energy = X_alphas_kop * Y_alphas_kop * partial_energies
    energy = torch.sum(energy, 1).view(bs, M)
    return energy.data


def get_candidates(X_mus, X_sigs, X_alphas,
                   Y_mus, Y_sigs, Y_alphas,
                   args, bs=100):
    """
    Get best translation pairs candidates.
    """

    all_scores = []
    all_targets = []

    # number of source words to consider
    N, K, D = X_mus.size()

    # nearest neighbors
    if args.dico_method == 'nn':

        # for every source word
        for i in range(0, N, bs):
            bidx_set = list(range(i, min(N, i + bs)))

            # compute target words scores
            scores = distdot_outer(X_mus[bidx_set], X_sigs[bidx_set], X_alphas[bidx_set],
                                   Y_mus, Y_sigs, Y_alphas)

            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    else:
        raise NotImplementedError('unknown dico_method {:s}'.format(args.dico_method))

    tmp1 = torch.arange(0, all_targets.size(0)).long().unsqueeze(1)
    tmp2 = all_targets[:, 0].long().unsqueeze(1)
    all_pairs = torch.cat([tmp1, tmp2], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (N, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if args.dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= args.dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if args.dico_max_size > 0:
        all_scores = all_scores[:args.dico_max_size]
        all_pairs = all_pairs[:args.dico_max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if args.dico_min_size > 0:
        diff[:args.dico_min_size] = 1e9

    # confidence threshold
    if args.dico_threshold > 0:
        mask = diff > args.dico_threshold
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs


def update_U(X_mus, X_sigs, X_alphas,
             Y_mus, Y_sigs, Y_alphas,
             mapping, args, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = 'S2T' in args.dico_build
    t2s = 'T2S' in args.dico_build
    assert s2t or t2s

    Xmus_new = mapping(X_mus)
    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(Xmus_new, X_sigs, X_alphas,
                                            Y_mus, Y_sigs, Y_alphas,
                                            args, bs=args.dico_bs)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(Y_mus, Y_sigs, Y_alphas,
                                            Xmus_new, X_sigs, X_alphas,
                                            args, bs=args.dico_bs)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if args.dico_build == 'S2T':
        dico = s2t_candidates
    elif args.dico_build == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates])
        if args.dico_build == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert args.dico_build == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                logger.warning("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[a, b] for (a, b) in final_pairs]))

    logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico.cuda() if args.cuda else dico

# X_mus: bs x K x D
# Y_mus: bs x K x D
# X_sigs: bs x K x 1
# Y_sigs: bs x K x 1
# X_alphas: bs x K
# Y_alphas: bs x K
def distdot_inner(X_mus, X_sigs, X_alphas,
                  Y_mus, Y_sigs, Y_alphas):
    bs, K, D = X_mus.size()
    # calcuate (X_mus - Y_mus)^T (X_mus - Y_mus)
    X_minus_Y = X_mus - Y_mus
    XY_dot = torch.bmm(X_minus_Y, X_minus_Y.permute(0, 2, 1))

    # calculate X_sigs + Y_sigs
    zero_mat = Variable(torch.zeros(K,K).cuda())
    XY_sig = torch.addbmm(zero_mat, X_sigs, Y_sigs.permute(0, 2, 1))

    # calculate xi: bs x K x K
    partial_energies = -0.5 * D * torch.log(XY_sig) -0.5 * D * np.log(np.pi)- 0.5 * (1. / XY_sig) * XY_dot
    partial_energies = torch.exp(partial_energies)

    # calculate energy
    XY_alphs = torch.bmm(X_alphas.unsqueeze(2), Y_alphas.unsqueeze(1))
    energy = XY_alphs * partial_energies
    energy = torch.sum(energy.view(bs, K*K), 1)
    return energy

def orthogonalize(mapping, map_beta=1e-3):
    if map_beta > 0:
        W = mapping.weight.data
        W.copy_((1 + map_beta) * W - map_beta * W.mm(W.transpose(0, 1).mm(W)))
    return mapping

def update_V(X_mus, X_sigs, X_alphas,
             Y_mus, Y_sigs, Y_alphas,
             dico, mapping, optimizer):

    Xmus_sub, Xsigs_sub, Xalphas_sub = X_mus[dico[:, 0]], X_sigs[dico[:, 0]], X_alphas[dico[:, 0]]
    Ymus_sub, Ysigs_sub, Yalphas_sub = Y_mus[dico[:, 1]], Y_sigs[dico[:, 1]], Y_alphas[dico[:, 1]]
    Xmus_sub = mapping(Xmus_sub)

    energy = distdot_inner(Xmus_sub, Xsigs_sub, Xalphas_sub,
                           Ymus_sub, Ysigs_sub, Yalphas_sub)

    loss = -torch.mean(energy)

    # optim
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mapping = orthogonalize(mapping)

    return mapping

def load_dictionary(X_dict, Y_dict, args):
    trn_dict_path = '%s/%s-%s/%s-%s.0-5000.txt' % (args.data, args.src, args.tgt, args.src, args.tgt)
    # tst_dict_path = '%s/%s-%s/%s-%s.5000-6500.txt' % (args.data, args.src, args.tgt, args.src, args.tgt)
    src_word2id = {v:k for k,v in X_dict['id2word'].iteritems()}
    tgt_word2id = {v:k for k,v in Y_dict['id2word'].iteritems()}

    trn_dict_pairs = []
    for line in open(trn_dict_path):
        word_x, word_y = line.strip().split()
        if word_x in src_word2id and word_y in tgt_word2id:
            trn_dict_pairs.append([src_word2id[word_x], tgt_word2id[word_y]])

    return np.random.permutation(trn_dict_pairs[:args.dico_init_size])

def main(args, logger):

    #args.cuda = args.gpu is not None
    #if args.cuda:
    #    torch.cuda.set_device(args.gpu)
    assert not args.cuda or torch.cuda.is_available()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --gpu")
        else:
            torch.cuda.manual_seed(args.rand_seed)

    # load word2gm model
    src_word2gm_modefile = "%s/%s-%s/%s.%s" % (args.data, args.src, args.tgt, args.src, args.emb_model)
    tgt_word2gm_modefile = "%s/%s-%s/%s.%s" % (args.data, args.src, args.tgt, args.tgt, args.emb_model)
    assert(os.path.isdir(src_word2gm_modefile))
    assert(os.path.isdir(tgt_word2gm_modefile))
    X_dict = load_w2gm_model(src_word2gm_modefile)
    Y_dict = load_w2gm_model(tgt_word2gm_modefile)

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
    # dico: (#pairs, 2) matrix
    # mapping: D x D orthogonal matrix
    mapping = nn.Linear(D, D, bias=False)
    mapping.weight.data.copy_(torch.from_numpy(ortho_group.rvs(D)).float())
    map_optimizer = torch.optim.SGD(mapping.parameters(), lr=args.map_lr)

    if args.cuda:
        X_mus = Variable(X_mus.cuda())
        Y_mus = Variable(Y_mus.cuda())
        X_sigs = Variable(X_sigs.cuda())
        Y_sigs = Variable(Y_sigs.cuda())
        X_alphas = Variable(X_alphas.cuda())
        Y_alphas = Variable(Y_alphas.cuda())
        mapping.cuda()

    # forward
    t = [0., 0.]
    try:
        for epoch in tqdm(range(1, args.epochs + 1)):

            start = time.time()

            # Update for U
            if epoch == 0 and args.dico_init:
                dico = load_dictionary(X_dict, Y_dict, args)
            else:
                dico = update_U(X_mus, X_sigs, X_alphas,
                                Y_mus, Y_sigs, Y_alphas,
                                mapping, args)
            start, elapsed = time.time(), (time.time() - start)
            t[0] += elapsed
            torch.save(dico, '%s/U.iter-%d.pth' % (args.log_dir, epoch))

            # Update for V
            mapping = update_V(X_mus, X_sigs, X_alphas,
                               Y_mus, Y_sigs, Y_alphas,
                               dico, mapping, map_optimizer)
            start, elapsed = time.time(), (time.time() - start)
            t[1] += elapsed
            torch.save(mapping.weight.data, '%s/V.iter-%d.pth' % (args.log_dir, epoch))
            exit(0)


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
