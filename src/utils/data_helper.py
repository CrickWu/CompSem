#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
USAGE:
OUTPUT:
"""

import torch
import os
import numpy as np
import random
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
from numpy import linalg as LA
from collections import defaultdict
from sklearn.metrics.pairwise import rbf_kernel

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def load_bi_dict(fname, splitter='\t'):
    bi_dict = defaultdict(list)
    for l in open(fname):
        ss = l.strip().split('\t')
        bi_dict[ss[0]].append(ss[1])
    return bi_dict

def load_text_vec(fname, splitter=' ', vocab_size=None):
    """
    Load dx1 word vecs from word2vec-like format:
    <word1> <dim1> <dim2> ...
    <word2> <dim1> <dim2> ...
    ...
    """
    word_vecs = defaultdict(list)
    words = []
    vecs = []
    with open(fname, "r") as f:
        if vocab_size is None:
            vocab_size = file_len(fname)
        layer1_size = None
        vocab_count = 0

        for line in tqdm(f.readlines()[:vocab_size+1]):
            ss = line.split(' ')
            if len(ss) <= 3:
                continue
            word = ss[0]
            dims = ' '.join(ss[1:]).strip().split(splitter)
            if layer1_size is None:
                layer1_size = len(dims)
                # print dims
                print("reading word2vec at vocab_size:%d, dimension:%d" % (vocab_size, layer1_size))

            vec = np.fromstring(' '.join(dims), dtype='float32', count=layer1_size, sep=' ')
            vec = vec/LA.norm(vec, 2) # normalize to unit length
            # vec = mean_center(vec) # dimension-wise mean centering
            word_vecs[word].append(vec)
            vecs.append(vec)
            words.append(word)
            vocab_count += 1
            if vocab_count >= vocab_size:
                break

        vecs = np.stack(vecs, axis=0)
        print('vecs.shape:', vecs.shape)
    return vocab_size, word_vecs, words, vecs, layer1_size

def downsample_frequent_words(counts, total_count, frequency_threshold=1e-3):
    if total_count > 1: # if inputs are counts
        threshold_count = float(frequency_threshold * total_count)
        probs = (np.sqrt(counts / threshold_count) + 1) * (threshold_count / counts)
        probs = np.maximum(probs, 1.0)    #Zm: Originally maximum, which upsamples rare words
        probs *= counts
        probs /= probs.sum()
    elif total_count <= 1: # inputs are frequency already
        probs = np.power(counts, 0.75)
        probs /= probs.sum()
    return probs



def load_freq(p, splitter=' '):
    """
    Load word frequence from word count
    <word1> <count1> ...
    <word2> <count2> ...
    ...
    return <word:word_count dictionary>, <freq_list>
    """
    w_count = defaultdict(list)
    count_list = []
    for l in open(p):
        ss = l.strip().split(splitter)
        w_count[ss[0]].append(float(ss[1]))
        count_list.append(float(ss[1]))

    counts = np.asarray(count_list, dtype=np.float32)
    total_count = counts.sum()
    return w_count, downsample_frequent_words(counts, total_count)

# utility function
def mean_center(matrix, axis=0):
    # print(type(matrix))
    if type(matrix) is np.ndarray:
        avg = np.mean(matrix, axis=axis, keepdims=True)
    else:
        avg = torch.mean(matrix, axis=axis, keepdims=True)
    return matrix - avg

def get_frequncy(filename, words):
    """
    format: <word> <count>
    Return a np.ndarray based on the order of `words`
    """
    word2freq = {}
    for num, line in enumerate(open(filename)):
        word, freq = line.split()
        word2freq[word] = float(freq)
    freq_vec = np.asarray([word2freq[word] for word in words])
    return freq_vec

def get_wordvec(filename):
    arr = []
    words = []
    for num, line in enumerate(open(filename)):
        if num == 0 and len(line.split()) < 4:
            continue
        arr.append(line.split()[1:])
        words.append(line.split()[0])
    # return np.vstack(arr).astype(float), words
    return mean_center(np.vstack(arr).astype(float), axis=0), words

def save_emb(filename, X, words):
    """
    save vectors X(np.array) and words to file f
    """
    f = open(filename, 'w')
    num_words = X.shape[0]
    assert num_words == len(words), "saved words is not aligned with saved vectors"
    f.write('{} {}\n'.format(num_words, X.shape[1]))
    for i,word in enumerate(words):
        word_vec_str = ' '.join(map(str, X[i,:].tolist()))
        f.write("{} {}\n".format(word, word_vec_str))

def get_dictionary_index(filename, src_words, tgt_words, limit=None, unique=False):
    src_idx, tgt_idx = [], []
    counter = 0
    for i,l in enumerate(open(filename).readlines()):
        if limit is not None and counter >= limit:
            break
        ss = l.strip().split()
        if ss[0].lower() in src_words and ss[1].lower() in tgt_words:
            if unique and (src_words.index(ss[0].lower()) in src_idx or tgt_words.index(ss[1].lower()) in tgt_idx):
                continue
            src_idx.append(src_words.index(ss[0].lower()))
            tgt_idx.append(tgt_words.index(ss[1].lower()))
            counter += 1
    return src_idx, tgt_idx

def get_dictionary_matrix(filename, src_words, tgt_words, limit=None, unique=False):
    """
    Suppose the dictionary has line format: <src_word> <tgt_word>
    """
    F = np.zeros((len(src_words),len(tgt_words)))
    src_idx, tgt_idx = get_dictionary_index(filename, src_words, tgt_words, limit=limit, unique=False)
    F[np.asarray(src_idx), np.asarray(tgt_idx)] = 1
    return F

def get_data(basedir, src, tgt):
    src_arr, src_words = get_wordvec(os.path.join(basedir, '{}-{}/'.format(src, tgt) + 'word2vec.' + src))
    tgt_arr, tgt_words = get_wordvec(os.path.join(basedir, '{}-{}/'.format(src, tgt) + 'word2vec.' + tgt))
    # src_arr, src_words = get_wordvec(os.path.join(basedir, src))
    # tgt_arr, tgt_words = get_wordvec(os.path.join(basedir, tgt))
    return src_arr, src_words, tgt_arr, tgt_words

def sparsify_mat(K, nn):
        ret_K = np.zeros(K.shape)
        for i in range(K.shape[0]):
            index = np.argsort(K[i, :])[-nn:]
            ret_K[i, index] = K[i, index]
        return ret_K

def sym_sparsify_mat(K, nn):
        K_sp = sparsify_mat(K, nn)
        K_sp = (K_sp + K_sp.T) / 2  # in case of non-positive semi-definite
        return K_sp

def get_adj_from_arr(src_arr, tgt_arr, nn, logger, normalize=True, kernel='cosine'):
    if normalize:
        src_arr = src_arr / np.linalg.norm(src_arr, ord=2, axis=1, keepdims=True)
        tgt_arr = tgt_arr / np.linalg.norm(tgt_arr, ord=2, axis=1, keepdims=True)
    if kernel == 'cosine':
        src_adj = src_arr.dot(src_arr.T)
        tgt_adj = tgt_arr.dot(tgt_arr.T)
    elif kernel == 'rbf':
        gamma = 1.0 / np.sqrt(src_arr.shape[1])
        src_adj = rbf_kernel(src_arr, gamma=gamma)
        tgt_adj = rbf_kernel(tgt_arr, gamma=gamma)
    if nn is not None:
        src_adj = sym_sparsify_mat(src_adj, nn)
        tgt_adj = sym_sparsify_mat(tgt_adj, nn)
        logger.info('Sparsification finished')
    else:
        logger.info('No sparsification')
    # print(type(src_adj), type(tgt_adj))
    return torch.from_numpy(src_adj.astype(float)), torch.from_numpy(tgt_adj.astype(float))

def get_adj(basedir, src, tgt, nn, logger, normalize=True, kernel='cosine'):
    logger.info('Loading data...')
    src_arr, src_words, tgt_arr, tgt_words = get_data(basedir, src, tgt)
    logger.info('Loading data finished')
    return get_adj_from_arr(src_arr, tgt_arr, nn, logger, normalize=normalize, kernel=kernel)


def mult_choice_F_data(args, m, n):
    if args.F_file is not None:  # past saved F_data file
        F_data = torch.from_numpy(np.loadtxt(args.F_file, dtype=float)).type(torch.FloatTensor)
    else:
        F_data = torch.rand(m, n) # non-negative

    if args.F_training is not None: # fill corresponding row with data
        Y, Y_words, X, X_words = get_data(args.data, args.src, args.tgt)
        F_trn = get_dictionary_matrix(args.F_training, Y_words, X_words, limit=args.train_size, unique=True)
        nonzero_row, nonzero_col = F_trn.nonzero()
        F_trn = torch.from_numpy(F_trn).float()
        train_set = set(nonzero_row)
        for row_idx in nonzero_row:
            F_data[row_idx,:] = F_trn[row_idx,:]
    else:
        train_set = set()

    F_data = F_data / F_data.norm(p=1, dim=1, keepdim=True) # row-sum == 1
    return F_data, train_set

if __name__ == '__main__':
    pass
    basedir = '../data/'
    src = 'es'
    tgt = 'en'
    nn = 5

    src_arr, tgt_arr = get_data(basedir, src, tgt)

    src_adj = sym_sparsify_mat(src_arr.dot(src_arr.T), nn)
    tgt_adj = sym_sparsify_mat(tgt_arr.dot(tgt_arr.T), nn)
    print(src_adj.shape)
    print(tgt_adj.shape)
    # print src_adj[0]