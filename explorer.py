#!/usr/bin/env python
from __future__ import print_function

import argparse
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

from byte import ByteCNN, UTF8Corpus, UTF8File
import logger as logger_module  # XXX
from logger import Logger


parser = argparse.ArgumentParser(description='Byte-level CNN text autoencoder.')
parser.add_argument('--resume-training', type=str, default='',
                    help='path to a training directory (loads the model and the optimizer)')
parser.add_argument('--resume-training-force-args', type=str, default='',
                    help='list of input args to be overwritten when resuming (e.g., # of epochs)')
args = parser.parse_args()


def setup_model(args):
    args, forced_args, state = logger_module.parse_resume_training(args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably "
                  "run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    dataset = UTF8Corpus(args.data, cuda=args.cuda)
    
    # Evaluate this early to know which data options to use
    model_kwargs = eval("dict(%s)" % (args.model_kwargs,))
    model = ByteCNN(**model_kwargs)
    
    if args.cuda:
        model.cuda()

    optimizer_proto = {'sgd': optim.SGD, 'adam': optim.Adam,
                       'adagrad': optim.Adagrad, 'adadelta': optim.Adadelta}
    optimizer_kwargs = eval("dict(%s)" % args.optimizer_kwargs)
    optimizer_kwargs['lr'] = args.lr
    optimizer = optimizer_proto[args.optimizer](
        model.parameters(), **optimizer_kwargs)
    
    # State has been loaded before model construction
    logger = state['logger']
    state = logger.set_training_state(state, optimizer)
    optimizer = state['optimizer']

    # if args.lr_lambda:
    #     # TODO Check how it behaves on resuming training
    #     lr_decay = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer, lr_lambda=eval(args.lr_lambda))
    # else:
    #     lr_decay = None
    
    if forced_args and forced_args.has_key('lr'):
        optimizer.param_groups[0]['lr'] = forced_args['lr']
        logger.lr = forced_args['lr']
    
    model.load_state_dict(logger.load_model_state_dict(current=True))
    first_epoch = logger.epoch + 1
    print(logger.logdir)

    return args, model, dataset, optimizer


args, model, dataset, optimizer = setup_model(args)

##############################################################################
# Decode sample sentences
##############################################################################

# Decode a sentence of my choice.
sentences = [
    "I'm so broke I can't even pay attention.",
    "One should always be in love. That is the reason one should never marry.",
    "We cannot do everything at once, but we can do something at once."]

for sent in sentences:
    print(repr(sent))
    print(model.try_on(dataset.valid.sample_batch(args.batch_size, sent),
                       switch_to_evalmode=False)[0])
    print('-----')

##############################################################################
# Decode to different length
##############################################################################

def try_on_varlen(model, batch_iterator):
    """Mimics model's try_on() on a range of target lengths."""
    model.train()
    decoded = {}
    for src in batch_iterator:
        src = Variable(src, volatile=True)
        src_r = model.encoder.num_recurrences(src)
	for r in range(4, src_r + 3):
            features = model.encoder(src, src_r)
            tgt = model.decoder(features, r)
            _, predictions = tgt.data.max(dim=1)

            # Make into strings and append to decoded
            for pred in predictions[:1]:
                pred = list(pred.cpu().numpy())
                pred = pred[:pred.index(UTF8File.EOS)] if UTF8File.EOS in pred else pred
                pred = repr(''.join([chr(c) for c in pred]))
                decoded[r] = pred
    return decoded

print('\n\n')
for sent in sentences:
    print(' '*4, repr(sent))
    decoded = try_on_varlen(model, dataset.valid.sample_batch(args.batch_size, sent))
    for r in sorted(decoded.keys()):
        print('{: <4}'.format(r), decoded[r])
    print('-----')
