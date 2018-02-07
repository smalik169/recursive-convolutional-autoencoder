from __future__ import print_function

import codecs
import pprint
import time
from collections import defaultdict
from itertools import chain

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class UTF8File(object):
    EOS = 0  # ASCII null symbol
    EMPTY = 7 # XXX
    def __init__(self, path, cuda, rng=None):
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)

        lines_by_len = defaultdict(list)
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                bytes_ = [ord(c) for c in line.strip().encode('utf-8')] + [self.EOS]
                bytes_ += [self.EMPTY] * (int(2 ** np.ceil(np.log2(len(bytes_)))) - len(bytes_))
                # Convnet reduces arbitrary length to 4
                if len(bytes_) < 4:
                    continue
                lines_by_len[len(bytes_)].append(bytes_)
        # Convert to ndarrays
        self.lines = {k: np.asarray(v, dtype=np.uint8) \
                      for k,v in lines_by_len.items()}

    def get_num_batches(self, bsz):
        return sum(arr.shape[0] // bsz for arr in self.lines.values())

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            for len_, data in self.lines.items():
                for batch in np.array_split(data, max(1, data.shape[0] // bsz)):
                    batch_tensor = torch.from_numpy(batch).long()
                    if self.cuda:
                        batch_tensor = batch_tensor.cuda()
                    yield (batch_tensor, batch_tensor) #(source, target)
        else:
            batch_inds = []
            for len_, data in self.lines.items():
                num_batches = data.shape[0] // bsz
                if num_batches == 0:
                    continue
                all_inds = np.random.permutation(data.shape[0])
                all_inds = all_inds[:(bsz * num_batches)]
                batch_inds += [(len_,inds) \
                               for inds in np.split(all_inds, num_batches)]
            np.random.shuffle(batch_inds)
            for len_, inds in batch_inds:
                batch_tensor = torch.from_numpy(self.lines[len_][inds]).long()
                if self.cuda:
                    batch_tensor = batch_tensor.cuda()
                yield (batch_tensor, batch_tensor) #(source, target)

    def sample_batch(self, bsz, sample_sentence=None):
        if not sample_sentence:
            sample_sentence = 'On a beautiful morning, a busty Amazon rode through a forest.'
        sample_sentence = sample_sentence.encode('utf-8')
        print("Source:", sample_sentence)
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [self.EOS] + \
                             [self.EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[1] == batch_len, bytes_.shape
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz)
        batch_tensor = torch.from_numpy(self.lines[batch_len][inds]).long()
        batch_tensor[0] = torch.from_numpy(bytes_).long()

        if self.cuda:
            batch_tensor = batch_tensor.cuda()
        yield (batch_tensor, batch_tensor) #(source, target)


class UTF8WordStarFile(object):
    EOS = 0  # ASCII null symbol
    EMPTY = 7 # XXX
    def __init__(self, path, cuda, rng=None, p=0.5):
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)

        lines_by_len = defaultdict(lambda: [list(), list()])
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                line = ' '.join(line.strip().encode('utf-8').split())
                len_ = int(2 ** np.ceil(np.log2(len(line))))
                source_bytes_ = []
                target_bytes_ = []
                for word in line.split():
                    bytes_ = [ord(c) for c in word] + [ord(' ')]
                    target_bytes_ += bytes_
                    if len(word) < 6 and np.random.random() < p:
                        source_bytes_ += [ord('*')] * len(word) + [ord(' ')]
                    else:
                        source_bytes_ += bytes_

                source_bytes_ = source_bytes_[:-1]
                source_bytes_ += [self.EMPTY] * (len_ - len(source_bytes_))
                target_bytes_ = target_bytes_[:-1]
                target_bytes_ += [self.EMPTY] * (len_ - len(target_bytes_))
                assert len(source_bytes_) == len(target_bytes_)
                # Convnet reduces arbitrary length to 4
                if len(source_bytes_) < 4:
                    continue
                lines_by_len[len(source_bytes_)][0].append(source_bytes_)
                lines_by_len[len(source_bytes_)][1].append(target_bytes_)
        # Convert to ndarrays
        self.lines = {k:
                (np.asarray(src, dtype=np.uint8),
                 np.asarray(tgt, dtype=np.uint8)) \
                    for k, (src, tgt) in lines_by_len.items()}

    def get_num_batches(self, bsz):
        return sum(arr.shape[0] // bsz for (arr, _) in self.lines.values())

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            for len_, (src_data, tgt_data) in self.lines.items():
                assert src_data.shape == tgt_data.shape
                for i in range(0, src_data.shape[0]-bsz, bsz):
                    src = torch.from_numpy(src_data[i: i+bsz]).long()
                    tgt = torch.from_numpy(tgt_data[i: i+bsz]).long()
                    yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)
        else:
            batch_inds = []
            for len_, (src_data, tgt_data) in self.lines.items():
                assert src_data.shape == tgt_data.shape
                num_batches = src_data.shape[0] // bsz
                if num_batches == 0:
                    continue
                all_inds = np.random.permutation(src_data.shape[0])
                all_inds = all_inds[:(bsz * num_batches)]
                batch_inds += [(len_,inds) \
                               for inds in np.split(all_inds, num_batches)]
            np.random.shuffle(batch_inds)
            for len_, inds in batch_inds:
                src = torch.from_numpy(self.lines[len_][0][inds]).long()
                tgt = torch.from_numpy(self.lines[len_][1][inds]).long()
                yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)

    def sample_batch(self, bsz, sample_sentence=None):
        if not sample_sentence:
            sample_sentence = 'On a beautiful morning, a busty Amazon rode through a forest.'
        sample_sentence = ' '.join(sample_sentence.strip().encode('utf-8').split())
        source_sentence = ' '.join(
                [('*' * len(word) if len(word) < 6 and np.random.random() < 0.5 else word)
                    for word in sample_sentence.split()])
        print("Source:", source_sentence)
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        source_bytes_ = [ord(c) for c in source_sentence] + [self.EOS]
        source_bytes_ += [self.EMPTY] * (batch_len - len(source_bytes_))
        source_bytes_ = np.asarray(source_bytes_)
        target_bytes_ = np.asarray(
                [[ord(c) for c in sample_sentence] + [self.EOS] + \
                [self.EMPTY] * (batch_len - len(sample_sentence) - 1)],
                dtype=np.uint8)
        assert source_bytes_.shape[0] == batch_len
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len][0]), bsz)
        src = torch.from_numpy(self.lines[batch_len][0][inds]).long()
        tgt = torch.from_numpy(self.lines[batch_len][1][inds]).long()
        src[0] = torch.from_numpy(source_bytes_).long()
        tgt[0] = torch.from_numpy(target_bytes_).long()
        yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)


class UTF8CharStarFile(object):
    EOS = 0  # ASCII null symbol
    EMPTY = 7 # XXX
    def __init__(self, path, cuda, rng=None, p=0.5):
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)
        self.p = p

        lines_by_len = defaultdict(list)
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                bytes_ = [ord(c) for c in line.strip().encode('utf-8')] + [self.EOS]
                bytes_ += [self.EMPTY] * (int(2 ** np.ceil(np.log2(len(bytes_)))) - len(bytes_))
                # Convnet reduces arbitrary length to 4
                if len(bytes_) < 4:
                    continue
                lines_by_len[len(bytes_)].append(bytes_)
        # Convert to ndarrays
        self.lines = {k: np.asarray(v, dtype=np.uint8) \
                      for k,v in lines_by_len.items()}

    def get_num_batches(self, bsz):
        return sum(arr.shape[0] // bsz for arr in self.lines.values())

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            for len_, data in self.lines.items():
                for batch in np.array_split(data, max(1, data.shape[0] // bsz)):
                    tgt = torch.from_numpy(batch).long()
                    src = tgt.clone()
                    mask = (torch.rand(src.size()) < self.p)
                    mask = mask & (src != self.EMPTY)
                    src[mask] = ord('*')
                    yield (src.cuda(), tgt.cuda) if self.cuda else (src, tgt)
        else:
            batch_inds = []
            for len_, data in self.lines.items():
                num_batches = data.shape[0] // bsz
                if num_batches == 0:
                    continue
                all_inds = np.random.permutation(data.shape[0])
                all_inds = all_inds[:(bsz * num_batches)]
                batch_inds += [(len_,inds) \
                               for inds in np.split(all_inds, num_batches)]
            np.random.shuffle(batch_inds)
            for len_, inds in batch_inds:
                tgt = torch.from_numpy(self.lines[len_][inds]).long()
                src = tgt.clone()
                mask = (torch.rand(src.size()) < self.p)
                mask = mask & (src != self.EMPTY)
                src[mask] = ord('*')
                yield (src.cuda(), tgt.cuda) if self.cuda else (src, tgt)

    def sample_batch(self, bsz, sample_sentence=None):
        if not sample_sentence:
            sample_sentence = 'On a beautiful morning, a busty Amazon rode through a forest.'
        sample_sentence = sample_sentence.encode('utf-8')
        print("Source:", sample_sentence)
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [self.EOS] + \
                             [self.EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[0] == batch_len
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz)
        tgt = torch.from_numpy(self.lines[batch_len][inds]).long()
        tgt[0] = torch.from_numpy(bytes_).long()
        src = tgt.clone()
        mask = (torch.rand(src.size()) < self.p)
        mask = mask & (src != self.EMPTY)
        src[mask] = ord('*')
        print("Source:", ''.join(map(chr, src[0].numpy()))
        yield (src.cuda(), tgt.cuda) if self.cuda else (src, tgt)


class UTF8Corpus(object):
    def __init__(self, path, cuda, file_class=UTF8File, rng=None):
        self.train = file_class(path + 'train.txt', cuda, rng=rng)
        self.valid = file_class(path + 'valid.txt', cuda, rng=rng)
        self.test = file_class(path + 'test.txt', cuda, rng=rng)