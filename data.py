from __future__ import print_function

import codecs
import os
import pprint
import time
from collections import defaultdict
from itertools import chain

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# TODO Make sure those symbols are absent in the data!
EOS = 0       # ASCII null symbol
EMPTY = 7     # ASCII bell
WILDCARD = 1  # ASCII start-of-heading (SOH)
SAMPLE_SENTENCE = 'On a beautiful morning, a busty Amazon rode through a forest.'

class Cache(object):
    @staticmethod
    def byte_file_to_lines(path, min_len=4, max_len=np.inf):
        lines_by_len = defaultdict(list)
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                bytes_ = [ord(c) for c in line.strip().encode('utf-8')] + [EOS]
                power2_len = int(np.ceil(np.log2(len(bytes_))))
                bytes_ += [EMPTY] * (2 ** power2_len - len(bytes_))
                # Convnet reduces arbitrary length to 4
                ### if len(bytes_) < 4:
                ###     continue
                lines_by_len[len(bytes_)].append(bytes_)
        # Convert to ndarrays
        for k in lines_by_len.keys():
            if min_len <= k <= max_len:
                lines_by_len[k] = np.asarray(lines_by_len[k], dtype=np.uint8)
            else:
                print('Dropping matrix of size %s: too long' % k)
                del lines_by_len[k]
        return lines_by_len

    @staticmethod
    def files(fpath):
        dir_ = os.path.dirname(fpath)
        base = os.path.basename(fpath)
        return [(f, os.path.join(dir_, f)) for f in os.listdir(dir_) \
                if f.startswith(base + '.len') and f.endswith('.uint8')]

    @staticmethod
    def build(fpath):
        if len(Cache.files(fpath)) > 0:
            return
        lines = Cache.byte_file_to_lines(fpath, max_len=np.inf)
        # Cache data matrices
        for k, v in lines.items():
            cached_path = path + ('.len%d.uint8' % k)
            if not os.path.isfile(cached_path):
                v.tofile(cached_path)

    @staticmethod
    def load(fpath, min_len=4, max_len=np.inf):
        lines = {}
        for name, path in Cache.files(fpath):
            key = int(name.split('.')[-2].replace('len', ''))
            val = np.fromfile(path, dtype=np.uint8).reshape(-1, key)
            if min_len <= val.shape[1] <= max_len:
                lines[key] = val
            else:
                print('Dropping matrix of size %s: too long' % str(val.shape))
        return lines


class RegularizedFile(object):

    def __init__(self, *args, **kwargs):
        self.utf8file = UTF8File(*args, **kwargs)
        self.random_file = RandomFile(*args, **kwargs)

    def iter_epoch(self, *args, **kwargs):
        it1 = self.utf8file.iter_epoch(*args, **kwargs)
        it2 = self.random_file.iter_epoch(*args, **kwargs)
        for a,b in zip(it1, it2):
            yield a
            yield b

    def get_num_batches(self, *args, **kwargs):
        return self.utf8file.get_num_batches(*args, **kwargs) + self.random_file.get_num_batches(*args, **kwargs)


class RandomFile(object):
    def __init__(self, path, cuda, rng=None, fixed_len=None, p=None,
                 use_cache=False, max_len=64, lowest_byte=32, highest_byte=122):
        if 'valid' in path or 'test' in path:
            self.num_samples = 10000
        elif 'train' in path:
            self.num_samples = 2 * 10**6
        else:
            raise ValueError
        del use_cache
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)
        self.fixed_len = fixed_len
        self.max_len = max_len
        self.lowest_byte = lowest_byte
        self.highest_byte = highest_byte

    def get_num_batches(self, bsz):
        return self.num_samples // bsz

    def maybe_pad(self, batch):
        if self.fixed_len:
            return np.pad(batch, ((0,0), (0, self.fixed_len - batch.shape[1])),
                          'constant', constant_values=EMPTY)
        else:
            return batch

    def iter_epoch(self, bsz, evaluation=False):
        is_power_of_2 = lambda x: 2**int(np.log2(x)) == x
        assert self.max_len is None or is_power_of_2(self.max_len)
        assert self.fixed_len is None or is_power_of_2(self.fixed_len)
        if self.fixed_len is not None:
            lengths = [self.fixed_len]
        else:
            log_len = int(np.log2(self.max_len))
            lengths = np.logspace(2, log_len, log_len-1, base=2)

        for _ in xrange(self.get_num_batches(bsz)):
            l = int(self.rng.choice(lengths))
            batch = self.rng.randint(
                self.lowest_byte, self.highest_byte+1, (bsz, l), dtype=np.uint8)
            batch_tensor = torch.from_numpy(batch).long()
            if self.cuda:
                batch_tensor = batch_tensor.cuda()
            yield (batch_tensor, batch_tensor) #(source, target)

    def sample_batch(self, bsz, sample_sentence=SAMPLE_SENTENCE):
        sample_sentence = sample_sentence.encode('utf-8')
        print("Source:", sample_sentence)
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        if self.fixed_len:
            batch_len = self.fixed_len
        elif self.max_len:
            batch_len = min(batch_len, self.max_len)

        if len(sample_sentence) > batch_len:
            sample_sentence = sample_sentence[:(batch_len-1)]

        batch = self.rng.randint(
            self.lowest_byte, self.highest_byte+1, (bsz, batch_len), dtype=np.uint8)
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [EOS] + \
                             [EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[1] == batch_len, bytes_.shape
        batch[0] = bytes_
        batch = self.maybe_pad(batch)
        batch_tensor = torch.from_numpy(batch).long()

        if self.cuda:
            batch_tensor = batch_tensor.cuda()
        yield (batch_tensor, batch_tensor) #(source, target)


class UTF8File(object):
    def __init__(self, path, cuda, rng=None, fixed_len=None, p=None,
                 min_len=4, max_len=np.inf, use_cache=True):
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)
        self.fixed_len = fixed_len
        self.min_len = min_len
        self.max_len = fixed_len or max_len
        if self.max_len is not None and self.fixed_len is not None:
            raise ValueError
        if use_cache:
            self.lines = Cache.load(path, min_len=min_len, max_len=self.max_len)
        else:
            self.lines = Cache.byte_file_to_lines(path, min_len=min_len, max_len=self.max_len)

    def get_num_batches(self, bsz):
        return sum(arr.shape[0] // bsz for arr in self.lines.values())

    def maybe_pad(self, batch):
        if self.fixed_len:
            return np.pad(batch, ((0,0), (0, self.fixed_len - batch.shape[1])),
                          'constant', constant_values=EMPTY)
        else:
            return batch

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            for len_, data in self.lines.items():
                for batch in np.array_split(data, max(1, data.shape[0] // bsz)):
                    batch = self.maybe_pad(batch)
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
                batch = self.lines[len_][inds]
                batch = self.maybe_pad(batch)
                batch_tensor = torch.from_numpy(batch).long()
                if self.cuda:
                    batch_tensor = batch_tensor.cuda()
                yield (batch_tensor, batch_tensor) #(source, target)

    def sample_batch(self, bsz, sample_sentence=SAMPLE_SENTENCE):
        sample_sentence = sample_sentence.encode('utf-8')
        print("Source:", sample_sentence)
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [EOS] + \
                             [EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[1] == batch_len, bytes_.shape
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz)
        batch = self.lines[batch_len][inds]
        batch[0] = bytes_
        batch = self.maybe_pad(batch)
        batch_tensor = torch.from_numpy(batch).long()

        if self.cuda:
            batch_tensor = batch_tensor.cuda()
        yield (batch_tensor, batch_tensor) #(source, target)


class UTF8WordStarFile(object):
    def __init__(self, path, cuda, rng=None, p=0.2, max_w_len=1000,
                 fixed_len=None, min_len=4, max_len=np.inf, use_cache=True):
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)
        self.p = p
        self.max_w_len = max_w_len

        lines_by_len = defaultdict(list)
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                bytes_ = [ord(c) for c in line.strip().encode('utf-8')] + [EOS]
                bytes_ += [EMPTY] * (int(2 ** np.ceil(np.log2(len(bytes_)))) - len(bytes_))
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
                    row[left:right] = WILDCARD
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

    def sample_batch(self, bsz, sample_sentence=SAMPLE_SENTENCE):
        sample_sentence = sample_sentence.encode('utf-8')
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [EOS] + \
                             [EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[1] == batch_len, bytes_.shape
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz)
        tgt = torch.from_numpy(self.lines[batch_len][inds]).long()
        tgt[0] = torch.from_numpy(bytes_).long()
        src = self._compy_and_mask_target(tgt)

        print("Source:",
              ''.join(map(chr, src[0].numpy())).replace(chr(WILDCARD), '*'))
        yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)


class UTF8CharStarFile(UTF8File):
    def __init__(self, path, cuda, p=0.2, **kwargs):
        super(UTF8CharStarFile, self).__init__(path, cuda, **kwargs)
        self.p = p

    def _get_mask(self, src):
        mask = (torch.rand(src.size()) < self.p)
        mask = mask & (src != EMPTY)
        return mask

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            for len_, data in self.lines.items():
                for batch in np.array_split(data, max(1, data.shape[0] // bsz)):
                    batch = self.maybe_pad(batch)
                    tgt = torch.from_numpy(batch).long()
                    src = tgt.clone()
                    mask = self._get_mask(src)
                    src[mask] = WILDCARD
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
                batch = self.lines[len_][inds]
                batch = self.maybe_pad(batch)
                tgt = torch.from_numpy(batch).long()
                src = tgt.clone()
                mask = self._get_mask(src)
                src[mask] = WILDCARD
                yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)

    def sample_batch(self, bsz, sample_sentence=SAMPLE_SENTENCE):
        sample_sentence = sample_sentence.encode('utf-8')
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [EOS] + \
                             [EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.uint8)
        assert bytes_.shape[1] == batch_len
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz)
        batch = self.lines[batch_len][inds]
        batch[0] = bytes_
        batch = self.maybe_pad(batch)
        tgt = torch.from_numpy(batch).long()
        src = tgt.clone()
        mask = (torch.rand(src.size()) < self.p)
        mask = mask & (src != EMPTY)
        src[mask] = WILDCARD
        print("Source:",
              ''.join(map(chr, src[0].numpy())).replace(chr(WILDCARD), '*'))
        yield (src.cuda(), tgt.cuda()) if self.cuda else (src, tgt)


class UTF8CharVarStarFile(UTF8CharStarFile):
    def _get_mask(self, src):
        # All probs drawn uniformly [0,p]
        sent_probs = torch.rand(src.size(0), 1) * self.p
        mask = (torch.rand(src.size()) < sent_probs)
        mask = mask & (src != EMPTY)
        return mask


class UTF8Corpus(object):
    def __init__(self, path, cuda, file_class=UTF8File, rng=None,
                 fixed_len=None, min_len=4, max_len=np.inf,
                 use_cache=True, sets=dict(train=True, valid=True, test=True)):
        if use_cache:
            Cache.build(path + 'train.txt')
            Cache.build(path + 'valid.txt')
            Cache.build(path + 'test.txt')
        subset_kwargs = dict(rng=rng, fixed_len=fixed_len,
            min_len=min_len, max_len=max_len, use_cache=use_cache)
        self.train = file_class(path + 'train.txt', cuda, **subset_kwargs) if sets['train'] else None
        self.valid = file_class(path + 'valid.txt', cuda, **subset_kwargs) if sets['valid'] else None
        self.test = file_class(path + 'test.txt', cuda, **subset_kwargs) if sets['test'] else None
