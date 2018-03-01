#!/usr/bin/env python
from __future__ import print_function

from collections import Counter, defaultdict

import numpy as np

import torch
from torch.autograd import Variable

import data
import models


def _header(h):
    return '\n'.join(['#' * 79, '# ' + h, '#' * 79])

def save_latent_codes(args, dataset, model, optimizer):
    print(_header('Save latent codes'))
    model.train()
    features = []
    for (src, _) in dataset.valid.iter_epoch(bsz=128, evaluation=False):
        src = Variable(src, volatile=True)
        r_src = model.num_recurrences(src)
        batch_features = model.encoder(src, r_src)
        features.append(batch_features.data.cpu().numpy())
    print(features[0].shape)
    features = np.vstack(features)
    print(features.shape)
    features.tofile('valid_features.npy')

def report_validation_accuracy(args, dataset, model, optimizer):
    print(_header('Report validation accuracy'))
    val_loss = model.eval_on(
        dataset.valid.iter_epoch(args.batch_size, evaluation=True),
        switch_to_evalmode=False)
    loss_str = ' : '.join(
        [' {} {:5.2f}'.format(k, v) for k, v in dict(val_loss).items()])
    loss_str = ('valid {}'.format(loss_str))
    print('\n', '-' * len(loss_str), loss_str, '-' * len(loss_str), '\n')

def interpolate_between_sentences(args, dataset, model, optimizer):
    print(_header('Interpolate between sentences'))
    
    # NOTE: Keed sent lengths within a pair fairly similar
    sent_pairs = [('I had a great time at the restaurant, the food was great!',
                   'I had an awful time at the diner, the meal was really bad.')]
    model.train()
    for (s1, s2) in sent_pairs:
        codes = []
        for sent in [s1, s2]:
            src, _ = dataset.valid.sample_batch(args.batch_size, sent).next()
            src = Variable(src, volatile=True)
            r_src = model.num_recurrences(src)
            if type(model) is models.VAEByteCNN:
                dist_params = model.projection(model.encoder(src, r_src))
                mu, _ = dist_params.chunk(2, dim=1)
                features = mu
            elif type(model) is models.ByteCNN:
                features = model.encoder(src, r_src)
            else:
                raise ValueError
            codes.append(features[0].data.cpu().numpy())
        num_codes = 8
        w = np.linspace(1, 0, num_codes)
        codes = w[:,None] * codes[0] + w[::-1,None] * codes[1]
        for c in codes:
            c = torch.from_numpy(c).cuda() if args.cuda else torch.from_numpy(c)
            # c = Variable(c)
            features[0] = c
            tgt = model.decoder(features, r_src)
            _, predictions = tgt.data.max(dim=1)
            pred = predictions[0]
            pred = list(pred.cpu().numpy())
            pred = pred[:pred.index(data.EOS)] if data.EOS in pred else pred
            pred = repr(''.join([chr(c) for c in pred]))
            print(pred) # decoded[r].append(pred)

def average_line_length(args, dataset, model, optimizer):
    print(_header('Average line length'))
    sum_ = 0
    lens = 0
    for subset in [dataset.train, dataset.valid, dataset.test]:
        for val in subset.lines.values():
            sum_ += np.sum(val != data.EMPTY)
            lens += val.shape[0]
    print('\nAverage sentence length (w/o padding): %.2f\n' % (1.0 * sum_ / lens))
   
def count_bytes_in_data(args, dataset, model, optimizer):
    print(_header('Count bytes in data'))
    byte_counter = Counter()
    for subset in [dataset.train, dataset.valid, dataset.test]:
        for val in subset.lines.values():
            byte_counter.update(dict(zip(*np.unique(val, return_counts=True))))
    print('')
    for k in range(256):
        print('{: >3}: {: >10}'.format(k, byte_counter.get(k, 0)))
    print('')
   
def decode_sample_sentences(args, dataset, model, optimizer):
    print(_header('Decode sample sentences'))
    
    # Decode a sentence of my choice.
    sentences = [
        "I'm so broke I can't even pay attention.",
        "One should always be in love. That is the reason one should never marry.",
        "We cannot do everything at once, but we can do something at once.",
        "Young man, in mathematics you don't understand things. You just get used to them.",
        "Szymon jedzie rowerem w dalekie trasy. Podziwia widoki i cieszy go szum wiatru."]
    
    for sent in sentences:
        print(repr(sent))
        print(model.try_on(dataset.valid.sample_batch(args.batch_size, sent),
                           switch_to_evalmode=False)[0])
        print('-----')
   
def decode_to_different_length(args, dataset, model, optimizer):
    print(_header('Decode to different length'))
    
    def try_on_varlen(model, batch_iterator):
        """Mimics model's try_on() on a range of target lengths."""
        model.train()
        decoded = {}
        for src, _ in batch_iterator:
            src = Variable(src, volatile=True)
            src_r = model.num_recurrences(src)
    	for r in range(4, src_r + 3):
                features = model.encoder(src, src_r)
                tgt = model.decoder(features, r)
                _, predictions = tgt.data.max(dim=1)
    
                # Make into strings and append to decoded
                for pred in predictions[:1]:
                    pred = list(pred.cpu().numpy())
                    pred = pred[:pred.index(data.EOS)] if data.EOS in pred else pred
                    pred = repr(''.join([chr(c) for c in pred]))
                    decoded[r] = pred
        return decoded
    
    print('\n\n')
    for sent in sentences:
        print(' '*4, repr(sent))
        for r in range(4, 9):
            decoded = model.try_on(dataset.valid.sample_batch(args.batch_size, sent),
                                   switch_to_evalmode=False, r_tgt=r)[0]
            print('{: <4}'.format(r), decoded)
        print('-----')
   
def decode_from_noise_vae(args, dataset, model, optimizer):
    print(_header('Decode from noise (VAE)'))
    print('\n\n')
    for _ in range(10):
        decoded = model.try_on(dataset.valid.sample_batch(args.batch_size),
                               switch_to_evalmode=False, first_sample_random=True)[0]
        print(decoded)
        print('-----')
   
def decode_from_noise(args, dataset, model, optimizer):
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
                pred = list(pred.cpu().numpy())
                pred = pred[:pred.index(data.EOS)] if data.EOS in pred else pred
                pred = repr(''.join([chr(c) for c in pred]))
                decoded[r].append(pred)
        return decoded
    
    print('\n\n')
    for noise_batch in [torch.randn([64, model.emsize*4]), 
            torch.randn([64, model.emsize*4])]:
        if args.cuda:
            noise_batch = noise_batch.cuda()
        decoded = try_on_sampledhid(model, noise_batch)
        for i in range(10):
            for r in sorted(decoded.keys()):
                print('{: <4}'.format(r), decoded[r][i])
            print('\n')
        print('\n')

def analyze(args, dataset, model, optimizer):
    save_latent_codes(args, dataset, model, optimizer)
    report_validation_accuracy(args, dataset, model, optimizer)
    interpolate_between_sentences(args, dataset, model, optimizer)
    average_line_length(args, dataset, model, optimizer)
    # count_bytes_in_data(args, dataset, model, optimizer)
    decode_sample_sentences(args, dataset, model, optimizer)
    decode_to_different_length(args, dataset, model, optimizer)
    decode_from_noise_vae(args, dataset, model, optimizer)
    decode_from_noise(args, dataset, model, optimizer)
    
    
