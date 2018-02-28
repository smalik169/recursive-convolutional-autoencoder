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
    WILDCARD = ord('*')
    def __init__(self, path, cuda, rng=None, p=0.5, max_w_len=6):
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)
        self.p = p
        self.max_w_len = max_w_len

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

    def _mask_row(self, row):
        def is_num_alpha(num):
            return ord('A') <= num <= ord('z')

        left, right = 0, 1
        _len = row.size(0)
        while right <= _len:
            if not is_num_alpha(row[left]):
                left += 1
                right = left
            elif right == _len or (not is_num_alpha(row[right])):
                if np.random.rand() < self.p and right - left <= self.max_w_len:
                    row[left:right] = self.WILDCARD
                left = right

            right += 1
        return row

    def _compy_and_mask_target(self, tgt):
        src = tgt.clone()
        for row in src:
            left, right = 0, 1
            row = self._mask_row(row)
        return src

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            for len_, data in self.lines.items():
                for batch in np.array_split(data, max(1, data.shape[0] // bsz)):
                    tgt = torch.from_numpy(batch).long()
                    src = self._compy_and_mask_target(tgt)
                    yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)
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
                src = self._compy_and_mask_target(tgt)
                yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)

    def sample_batch(self, bsz, sample_sentence=None):
        if not sample_sentence:
            sample_sentence = 'On a beautiful morning, a busty Amazon rode through a forest.'
        sample_sentence = sample_sentence.encode('utf-8')
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [self.EOS] + \
                             [self.EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[1] == batch_len, bytes_.shape
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz)
        tgt = torch.from_numpy(self.lines[batch_len][inds]).long()
        tgt[0] = torch.from_numpy(bytes_).long()
        src = self._compy_and_mask_target(tgt)

        print("Source:", ''.join(map(chr, src[0].numpy())))
        yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)


class UTF8CharStarFile(object):
    EOS = 0  # ASCII null symbol
    EMPTY = 7 # XXX
    WILDCARD = ord('*')
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

    def _get_mask(self, src):
        mask = (torch.rand(src.size()) < self.p)
        mask = mask & (src != self.EMPTY)
        return mask

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            for len_, data in self.lines.items():
                for batch in np.array_split(data, max(1, data.shape[0] // bsz)):
                    tgt = torch.from_numpy(batch).long()
                    src = tgt.clone()
                    mask = self._get_mask(src)
                    src[mask] = self.WILDCARD
                    yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)
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
                mask = self._get_mask(src)
                src[mask] = self.WILDCARD
                yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)

    def sample_batch(self, bsz, sample_sentence=None):
        if not sample_sentence:
            sample_sentence = 'On a beautiful morning, a busty Amazon rode through a forest.'
        sample_sentence = sample_sentence.encode('utf-8')
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [self.EOS] + \
                             [self.EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[1] == batch_len
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz)
        tgt = torch.from_numpy(self.lines[batch_len][inds]).long()
        tgt[0] = torch.from_numpy(bytes_).long()
        src = tgt.clone()
        mask = (torch.rand(src.size()) < self.p)
        mask = mask & (src != self.EMPTY)
        src[mask] = ord('*')
        print("Source:", ''.join(map(chr, src[0].numpy())))
        yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)


class UTF8CharVarStarFile(UTF8CharStarFile):
    WILDCARD = 1  # ASCII start-of-heading (SOH)
    def _get_mask(self, src):
        # Half has probs p, half probs drawn uniformly [0,p]
        bsz = src.size(0)
        sent_probs = torch.cat([torch.rand(bsz // 2, 1) * self.p,
                                torch.ones(bsz - (bsz // 2), 1) * self.p],
                               dim=0)
        mask = (torch.rand(src.size()) < sent_probs)
        mask = mask & (src != self.EMPTY)
        return mask

class UTF8Corpus(object):
    def __init__(self, path, cuda, file_class=UTF8File, rng=None):
        self.train = file_class(path + 'train.txt', cuda, rng=rng)
        self.valid = file_class(path + 'valid.txt', cuda, rng=rng)
        self.test = file_class(path + 'test.txt', cuda, rng=rng)
