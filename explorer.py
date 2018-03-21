#!/usr/bin/env python
from __future__ import print_function

import codecs
import copy
import cPickle
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import torch
from torch.autograd import Variable

import data
import models


amazon_sent = 'On a beautiful morning, a busty Amazon rode through a forest.'

wildcarded_sentences = [
    "I'm so broke I can't even * attention.",
    "I'm so broke I can't even ** attention.",
    "I'm so broke I can't even *** attention.",
    "I'm so broke I can't even **** attention.",
    "I'm so broke I can't even ***** attention.",
    "I liked the meal, it was very *** and delicious.",
    "I liked the meal, it was very **** and delicious.",
    "I liked the meal, it was very ***** and delicious.",
    "I liked the meal, it was very ****** and delicious.",
    "I liked the meal, it was very ******* and delicious.",
    "I liked the meal, it was very ******** and delicious.",
    "One * always be in love. That is the reason one should never marry.",
    "One ** always be in love. That is the reason one should never marry.",
    "One *** always be in love. That is the reason one should never marry.",
    "One **** always be in love. That is the reason one should never marry.",
    "One ***** always be in love. That is the reason one should never marry.",
    "Big ****** taxi cab.",
    "Cat ***** dog.",
    "You're ******* my friend.",
    "I'm so ***** I can't even pay attention.",
    "I'm so broke I can't even *** attention.",
    "One ****** always be in love. That is the reason one should never marry.",
    "One should always be in ****. That is the reason one should never marry.",
    "One should always be in love. That is the ****** one should never marry.",
    "One should always be in love. That is the reason one should never ***** .",
    "We cannot do everything at once, but we can do something at once.",
    "Young man, in mathematics you don't understand things. You just get used to them.",
]

sentences = [
    "I'm so broke I can't even pay attention.",
    "One should always be in love. That is the reason one should never marry.",
    "We cannot do everything at once, but we can do something at once.",
    "Young man, in mathematics you don't understand things. You just get used to them.",
    "Szymon jedzie rowerem w dalekie trasy. Podziwia widoki i cieszy go szum wiatru.",
    # "We cannot do everything at once, but we can do something at once. We cannot do everything at once, but we can do something at once.",
    # "Young man, in mathematics you don't understand things. You just get used to them. Young man, in mathematics you don't understand things. You just get used to them.",
    "On a reautiful worning, a baste (tadan lose through a corest.",
]

long_sentences = [
    "However, the youth died suddenly and suspiciously on 12 February 55, the very day before his proclamation as an adult had been set.",
    "In 1943 a British Court of Inquiry investigated the crash of Sikorski's Liberator II AL523, but was unable to determine the probable cause, finding only that it was an accident and the \"aircraft became uncontrollable for reasons which cannot be established\".",
    "Although superior in numbers, Dokhturov's column had no supporting artillery, and the narrow space prevented them from taking advantage of their size.",
    "Alfredo Lim once again ran for mayor and defeated Atienza's son Ali in the 2007 city election and immediately reversed all of Atienza's projects claiming Atienza's projects made little contribution to the improvements of the city.",
    "Boats of the Type UB I design were built by two manufacturers, Germaniawerft of Kiel and AG Weser of Bremen, which led to some variations in boats from the two shipyards.",
    "Olmec colossal heads were fashioned as in-the-round monuments with varying levels of relief on the same work; they tended to feature higher relief on the face and lower relief on the earspools and headdresses.",
    "Nerva had seen the anarchy which had resulted from the death of Nero; he knew that to hesitate even for a few hours could lead to violent civil conflict.",
    "It caused enormous disruption to Chinese society: the census of 754 recorded 52.9 million people, but ten years later, the census counted just 16.9 million, the remainder having been displaced or killed.",
    "It was designed by series production designer Michael Pickwood, who stated that the new interior was also supposed to be \"darker and moodier\" and provide an easier access to the \"gallery\" of the ship when shooting.",
    "Due to its track away from landmasses, damage remained minimal; however, as Nangka passed to the south and east of Japan, the storm brought light rainfall to the country, peaking at 81 mm (3.2 in) in Minamidaito, Okinawa.",
]

weird_sentences = [
    " "*31,
    "in "*8,
    "in  "*8,
    "in   "*6,
    "Phoebe",
    "Phoebe has",
    "Phoebe has cats",
    "Catscatscatscatscats",
    "Catscatscatscatscats.",
    "cats cats cats cats cats",
    "Cats cats cats cats cats",
]

sent_pairs = [('I had a great time at the restaurant, the food was great!',
               'I had an awful time at the diner, the meal was really bad.')]


def _header(h):
    return '#  ' + h

def _to_bytes(u, add_eos=False):
    ret = [ord(c) for c in u.encode('utf-8')]
    return ret + [data.EOS] if add_eos else ret

def _to_unicode(bytes_):
    return ''.join([chr(c) for c in bytes_]).decode('utf-8')

def to_np(x):
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy()

def from_np(x, cuda):
    return torch.from_numpy(x).cuda() if cuda else torch.from_numpy(x)

def to_variable(x, cuda, **kwargs):
    x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    return Variable(x, **kwargs)


class Vocab(object):

    def __init__(self, path, use_cache=True):
        self.use_cache = use_cache
        self.words = set()
        self._load_vocab(path, use_cache)

    def _vocab_path(self, path):
        return os.path.basename(path) + 'vocab.pkl'

    def _load_vocab(self, path, use_cache):
   
        if self.use_cache and os.path.isfile(self._vocab_path(path)):
            self.words = cPickle.load(open(self._vocab_path(path), 'rb'))
            return
    
        counter = Counter()
        for sset in ('train.txt', 'valid.txt', 'test.txt'):
            with codecs.open(path + sset, 'r', 'utf-8') as f:
                for line in f:
                    counter.update(line.strip().split())
        print('Loaded vocab of size %d' % len(counter))
        for k,v in counter.items():
            if v > 3:
                self.words.update([k])
        print('Loaded vocab of size %d' % len(self.words))

        if self.use_cache:
            cPickle.dump(self.words, open(self._vocab_path(path), 'wb'))

    def most_probable(self, letter_probs):
        len_ = letter_probs.shape[0]
        scored = []
        for w in self.words:
            if len(w.encode('utf-8')) != len_:
                continue
            # if not all(ord('a') <= c <= ord('z') for c in _to_bytes(v)):
            #     continue
            probsum = np.sum(letter_probs[i, c] for (i,c) in enumerate(_to_bytes(w)))
            scored.append((w, probsum))
        return sorted(scored, key=lambda p: -p[1])

    def most_probable_pair(self, letter_probs):
        raise NotImplementedError


class Explorer(object):

    def __init__(self, args, dataset, model, optimizer, logger):
        self.args = args
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.logger = logger

        self.vocab = None

    def analyze(self):
        # for l in [16, 32, 64, 128, 256, 512]:
        #     self.validation_accuracy(min_len=l, max_len=l)
        self.integrated_gradients(aim='output', sent=long_sentences[8])
    
        # self.decode_from_noise_vae()
        # self.integrated_gradients(aim='latent')
        # self.fill_vocab_word(sents=wildcarded_sentences)
    
        # self.decode_sample_sentences()
        # self.validation_accuracy()
        # self.decode_sample_sentences(sents=weird_sentences, show_errors=True)
        # self.decode_sample_sentences(sents=long_sentences, show_errors=True)
    
        # self.save_latent_codes()
        # self.interpolate_between_sentences()
        # self.average_line_length()
        # self.count_bytes_in_data()
        # self.decode_to_different_length()
        # self.decode_from_noise()

    def integrated_gradients(self, aim='output', integration_points=50,
                             sent=amazon_sent):
        self.model.train()  # XXX BatchNorm
    
        # Contruct a "fake" batch
        batch_size = 1 if self.model.instance_norm else self.args.batch_size
        src, _ = self.dataset.valid.sample_batch(
            batch_size, sample_sentence=sent).next()
        src = Variable(src)
        r = self.model.num_recurrences(src)
        x = self.model.encoder.embedding(src).transpose(1, 2).data
    
        # Contruct a "neutral" reference batch
        mean_emb = self.model.encoder.embedding.weight.data.mean(dim=1)
        x_mean = x.clone()
        x_mean[0] = mean_emb.unsqueeze(1).expand(x_mean.size(1),
                                                 x_mean.size(2))
        # for j in range(x_mean.size(2)):
        #     x_mean[0,:,j] = mean_emb

        if aim == 'output':
            def get_features_fun(src_emb):
                latent = self.model.encoder(src_emb, r=r, embed=False)
                return self.model.decoder(latent, r=r)[0]
        elif aim == 'latent':
            def get_features_fun(src_emb):
                latent = self.model.encoder(src_emb, r=r, embed=False)[0]
                return torch.abs(latent)
        else:
            raise ValueError('Unkown aim for integrated gradients')

        grad_norms = None  # Don't know feature.size() yet, allocate later
        grad_fname = '%s_integr_grads_%s' % (self.logger.logdir[:-1], aim)
        sent_len = x.size(2)  # This len is padded
        weights = np.linspace(0.0, 1.0, integration_points).astype('float32')
    
        for i in range(0, integration_points):
            sys.stdout.write('Integration point %d/%d ' % (i+1, integration_points))
           
            # Mix x and x_mean using weights
            batch = weights[i] * x + (1.0 - weights[i]) * x_mean
            batch = Variable(batch.cuda() if self.args.cuda else batch, requires_grad=True)
    
            feats = get_features_fun(batch)
            num_feats = feats.size(1)
    
            if grad_norms is None:
                grad_norms = np.zeros((num_feats, sent_len, integration_points))
    
            for pos in range(num_feats):
                sys.stdout.write('.')
                sys.stdout.flush()
                truth_byte = src.data[0][pos]
                val = feats[truth_byte, pos]
                # grad size is (bsz x emsize x len)
                grad = torch.autograd.grad([val], [batch], retain_graph=True)[0]
                grad_norms[pos, :, i] = torch.norm(grad[0], p=2, dim=0)
            print('')
    
        np.save(grad_fname, grad_norms)
        print('Saved.')
        return grad_norms

    def save_latent_codes(self):
        print(_header('Save latent codes'))
        self.model.train()
        features = []
        for (src, _) in self.dataset.valid.iter_epoch(bsz=128, evaluation=False):
            src = Variable(src, volatile=True)
            r_src = self.model.num_recurrences(src)
            batch_features = self.model.encoder(src, r_src)
            features.append(to_np(batch_features))
        print(features[0].shape)
        features = np.vstack(features)
        print(features.shape)
        features.tofile('valid_features.npy')
    
    def validation_accuracy(self, min_len=None, max_len=None, header=True):

        assert (min_len is None and max_len is None) or \
            (min_len is not None and max_len is not None)

        if min_len:
            if header:
                print(_header('Validation accuracy (len %d to %d)' % (
                    min_len, max_len)))
            data_kwargs = eval('dict(%s)' % self.args.data_kwargs)
            data_kwargs.update(dict(min_len=min_len, max_len=max_len,
                                    sets=dict(train=False, valid=True, test=False)))
            dataset = data.UTF8Corpus(
                self.args.data, cuda=self.args.cuda,
                file_class=getattr(data, self.args.file_class), **data_kwargs)
        else:
            print(_header('Validation accuracy'))
            dataset = self.dataset

        val_loss = self.model.eval_on(
            dataset.valid.iter_epoch(self.args.batch_size, evaluation=True),
            switch_to_evalmode=False)
        loss_str = ' : '.join(
            [' {} {:5.2f}'.format(k, v) for k, v in dict(val_loss).items()])
        loss_str = ('valid {}'.format(loss_str))
        print('\n', '-' * len(loss_str), loss_str, '-' * len(loss_str), '\n')
    
    def interpolate_between_sentences(self):
        print(_header('Interpolate between sentences'))
        
        # NOTE: Keed sent lengths within a pair fairly similar
        self.model.train()
        for (s1, s2) in sent_pairs:
            codes = []
            for sent in [s1, s2]:
                src, _ = self.dataset.valid.sample_batch(self.args.batch_size, sent).next()
                src = Variable(src, volatile=True)
                r_src = self.model.num_recurrences(src)
                if type(self.model) is models.VAEByteCNN:
                    dist_params = self.model.projection(
                        self.model.encoder(src, r_src))
                    mu, _ = dist_params.chunk(2, dim=1)
                    features = mu
                elif type(self.model) is models.ByteCNN:
                    features = self.model.encoder(src, r_src)
                else:
                    raise ValueError
                codes.append(to_np(features[0]))
            num_codes = 8
            w = np.linspace(1, 0, num_codes)
            codes = w[:,None] * codes[0] + w[::-1,None] * codes[1]
            for code in codes:
                features[0] = from_np(code)
                tgt = self.model.decoder(features, r_src)
                _, predictions = tgt.data.max(dim=1)
                pred = predictions[0]
                pred = list(to_np(pred))
                pred = pred[:pred.index(data.EOS)] if data.EOS in pred else pred
                pred = repr(''.join([chr(c) for c in pred]))
                print(pred) # decoded[r].append(pred)
    
    def average_line_length(self):
        print(_header('Average line length'))
        sum_ = 0
        lens = 0
        for s in ('train', 'valid', 'test'):
            for val in getattr(self.dataset, 's').lines.values():
                sum_ += np.sum(val != data.EMPTY)
                lens += val.shape[0]
        print('\nAverage sentence length (w/o padding): %.2f\n' % (
            1.0 * sum_ / lens))
       
    def count_bytes_in_data(self):
        print(_header('Count bytes in data'))
        byte_counter = Counter()
        for s in ('train', 'valid', 'test'):
            for val in getattr(self.dataset, 's').lines.values():
                byte_counter.update(dict(zip(*np.unique(val, return_counts=True))))
        print('')
        for k in range(256):
            print('{: >3}: {: >10}'.format(k, byte_counter.get(k, 0)))
        print('')
       
    def decode_sample_sentences(self, sents=None, show_errors=False):
        print(_header('Decode sample sentences'))
        
        if not sents:
            sents = sentences
        
        for sent in sents:
            src = repr(sent)
            tgt = repr(self.model.try_on(self.dataset.valid.sample_batch(self.args.batch_size, sent),
                                   switch_to_evalmode=False)[0])
            mask = None
            if show_errors:
                mask = ''.join([' ' if a == b else 'E' for a,b in zip(src, tgt)])
            for i in range(0, len(src), 128):
                print(src[i:i+128])
                print(tgt[i:i+128])
                if mask:
                    print(mask[i:i+128])
            print('-----')
       
    def decode_to_different_length(self):
        print(_header('Decode to different length'))
        for sent in sentences:
            print(' '*4, repr(sent))
            for r in range(4, 9):
                decoded = repr(self.model.try_on(self.dataset.valid.sample_batch(self.args.batch_size, sent),
                                       switch_to_evalmode=False, r_tgt=r)[0])
                print('{: <4}'.format(r), decoded)
            print('-----')
       
    def decode_from_noise_vae(self):
        print(_header('Decode from noise (VAE)'))
        print('\n\n')
        for _ in range(10):
            predicted, outputs = self.model.try_on(self.dataset.valid.sample_batch(self.args.batch_size, sample_sentence=(' '*127)),
                                              switch_to_evalmode=False, first_sample_random=True, return_outputs=True)
            # Strip quote chars from repr(predicted) with [1:-1]
            predicted, outputs = predicted[0], outputs[0][0]
            vocabularize(self.args, outputs, predicted, wildcarded=False)
            print('-----')
       
    def decode_from_noise(self):
        print(_header('Decode from noise'))
        
        def try_on_sampledhid(model, hid):
            """Mimics model's try_on() on a range of target lengths."""
            model.train()
            decoded = {}
            hid = Variable(hid, volatile=True)
            for r in range(4, 8):
                decoded[r] = []
                tgt = model.decoder(hid, r)
                _, predictions = tgt.data.max(dim=1)
        
                # Make into strings and append to decoded
                for pred in predictions:
                    pred = list(to_np(pred))
                    pred = pred[:pred.index(data.EOS)] if data.EOS in pred else pred
                    pred = repr(''.join([chr(c) for c in pred]))
                    decoded[r].append(pred)
            return decoded
        
        print('\n\n')
        for noise_batch in [torch.randn([64, self.model.emsize*4]), 
                torch.randn([64, self.model.emsize*4])]:
            if self.args.cuda:
                noise_batch = noise_batch.cuda()
            decoded = try_on_sampledhid(self.model, noise_batch)
            for i in range(10):
                for r in sorted(decoded.keys()):
                    print('{: <4}'.format(r), decoded[r][i])
                print('\n')
            print('\n')
    
    def vocabularize(self, outputs, decoded, wildcarded=False, nsamples=5):
    
        if self.vocab is None:
            self.vocab = Vocab(self.args.data)
    
        if type(decoded) is str:
            try:
                decoded = decoded.decode('utf-8')
            except Exception, e:
                print(str(e))
                return
        bytes_ = _to_bytes(decoded)
        orig_decoded = copy.deepcopy(decoded)
        orig_bytes = copy.deepcopy(bytes_)
    
        samples = [list(copy.deepcopy(orig_decoded).encode('utf-8')) \
                   for _ in xrange(nsamples)]
    
        if not wildcarded:
            for i, c in enumerate(bytes_):
                if not c in (ord('.'), ord(','), ord(' ')):
                    bytes_[i] = ord('*')
            decoded = _to_unicode(bytes_)
    
        print(repr(orig_decoded))
        print(repr(decoded))
    
        for i, c in enumerate(bytes_):
            if c == ord('*'):
                bytes_[i] = data.WILDCARD
    
        # Iterate over words with asterisks
        wild_start = [i for i in xrange(len(bytes_)) \
                      if bytes_[i] == data.WILDCARD and (i == 0 or bytes_[i-1] != data.WILDCARD)]
        wildcards = []
        for ws in wild_start:
            we = ws
            while we+1 < len(bytes_) and bytes_[we+1] == data.WILDCARD:
                we += 1
            wildcards.append((ws, we))
    
        for (ws, we) in wildcards:
            # Estimate probability of each word in vocab
            len_ = we - ws + 1
            true_word = _to_unicode(orig_bytes[ws:we+1])
            true_probsum = np.sum(outputs[orig_bytes[ws+i], ws+i] \
                                  for i in xrange(len_))
            # XXX To be really fair, outputs should be normalized
            # XXX to probability distributions, and true_probsum
            # XXX should be a sum of logarithms
    
            scored = self.vocab.most_probable(outputs[:, ws:we+1].T)
            if scored == []:
                unk = list('<unk>' + ' '*len_)
                for i in xrange(nsamples):
                    samples[i][ws:we+1] = unk[:len_]
            else:
                for i in xrange(nsamples):
                    samples[i][ws:we+1] = list((scored[i][0]).encode('utf-8'))
    
        for s in samples:
            print(repr(''.join(s).decode('utf-8')))
        print()
