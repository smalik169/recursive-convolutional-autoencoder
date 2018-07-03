import os

import numpy as np

import torch
import torch.nn as nn


class WordGloveBatcher(object):
    def __init__(self, glove_path, fixed_len=None, balance=False):
        self.glove_path = glove_path
        self.fixed_len = fixed_len
        self.balance = balance
        self.glove_vectors = {}

    def maybe_pad(self, sent, sent_len):
        if self.fixed_len is not None:
            sent = sent[:, :self.fixed_len]
            sent_len = [min(sl, self.fixed_len) for sl in sent_len]

        bsz, max_len, dim = sent.size()

        if (self.fixed_len is not None) and self.balance:
            batch = sent.new_zeros(bsz, self.fixed_len, dim)

            for i in range(bsz):
                block_size = self.fixed_len // sent_len[i]
                for j in range(min(sent_len[i], self.fixed_len)):
                    batch[i, j * block_size, :] = sent[i, j, :]
        else:
            power2_len = self.fixed_len or 2 ** int(np.ceil(np.log2(max_len)))
            if power2_len > max_len:
                sent_zero_pad = sent.new_zeros(
                        bsz, (power2_len - max_len), dim)
                batch = torch.cat([sent, sent_zero_pad], dim=1)
            else:
                batch = sent

        return batch, sent_len

    def build_vocab(self, sentences, tokenize=False):
        words = set(['<s>', '</s>']).union(*[
            set(word_tokenize(s) if tokenize else sent.split())
            for sent in sentences])

        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in words:
                    self.glove_vectors[word] = np.fromstring(vec, sep=' ')

            self.emsize = len(vec.split())

        assert self.emsize == 300 #XXX

    def prepare_batch(self, sentences, sent_len):
        max_len = max(sent_len)
        sent = np.zeros((len(sentences), max_len, self.emsize))

        for i in range(len(sentences)):
            for j in range(sent_len[i]):
                sent[i, j, :] = self.glove_vectors[sentences[i][j]]

        sent = torch.FloatTensor(sent)
        return self.maybe_pad(sent, sent_len)

    def prepare_samples(self, sentences, tokenize=False):
        sentences = [['<s>'] + word_tokenize(s) + ['</s>'] if tokenize else
                     ['<s>'] + s.split() + ['</s>'] for s in sentences]

        # filterout words without glove vectors
        for i in range(len(sentences)):
            sentences[i] = [
                    word for word in sentences[i] if word in self.glove_vectors
                    ] or ['</s>']

        lengths = np.array([len(sent) for sent in sentences])
        sentences = np.array(sentences)
        return sentences, lengths


class NLIWordData(object):
    def __init__(self, s1_path, s2_path, labels_path, cuda):
        self.cuda = cuda
        with open(s1_path, 'r') as f:
            self.s1 = [line.strip() for line in f]
        with open(s2_path, 'r') as f:
            self.s2 = [line.strip() for line in f]
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f]

        unique_labels = sorted(set(labels))
        num_labels = len(unique_labels)
        assert num_labels == 3
        labels_dict = {unique_labels[i]: i for i in range(num_labels)}
        self.labels = np.array([labels_dict[l] for l in labels])

    def add_batcher(self, batcher):
        self.batcher = batcher

    def get_num_batches(self, bsz):
        return int(np.ceil(self.num_sentences / bsz))

    def prepare_samples(self):
        self.sentences1, self.lengths1 = self.batcher.prepare_samples(self.s1)
        del self.s1

        self.sentences2, self.lengths2 = self.batcher.prepare_samples(self.s2)
        del self.s2

        self.num_sentences = self.sentences1.shape[0]
        assert self.sentences1.shape[0] == self.sentences2.shape[0]

    def iter_epoch(self, bsz, evaluation=False):
        if evaluation:
            permutation = np.arange(self.num_sentences)
        else:
            permutation = np.random.permutation(self.num_sentences)

        for i in range(0, self.num_sentences, bsz):
            batch_inds = permutation[i:i+bsz]
            sent1, lens1 = self.batcher.prepare_batch(
                    self.sentences1[batch_inds], self.lengths1[batch_inds])

            sent2, lens2 = self.batcher.prepare_batch(
                    self.sentences2[batch_inds], self.lengths2[batch_inds])

            labels = torch.from_numpy(self.labels[batch_inds]).long()
            if self.cuda:
                sent1 = sent1.cuda()
                sent2 = sent2.cuda()
                labels = labels.cuda()

            yield (sent1, lens1), (sent2, lens2), labels


class NLIWordCorpus(object):
    def __init__(self, data_path, glove_path, cuda, fixed_len=None, balance=False):
        self.train = NLIWordData(
                os.path.join(data_path, "s1.train"),
                os.path.join(data_path, "s2.train"),
                os.path.join(data_path, "labels.train"),
                cuda)
        self.valid = NLIWordData(
                os.path.join(data_path, "s1.dev"),
                os.path.join(data_path, "s2.dev"),
                os.path.join(data_path, "labels.dev"),
                cuda)
        self.test = NLIWordData(
                os.path.join(data_path, "s1.test"),
                os.path.join(data_path, "s2.test"),
                os.path.join(data_path, "labels.test"),
                cuda)

        self.batcher = WordGloveBatcher(glove_path, fixed_len, balance)
        self.batcher.build_vocab(
                [s for dataset in [self.train, self.valid, self.test]
                   for sent in ["s1", "s2"]
                   for s in getattr(dataset, sent)])

        self.train.add_batcher(self.batcher)
        self.valid.add_batcher(self.batcher)
        self.test.add_batcher(self.batcher)

        self.train.prepare_samples()
        self.valid.prepare_samples()
        self.test.prepare_samples()
