#!/usr/bin/env python
from __future__ import print_function

import argparse
import codecs
import os
import pprint
import sys
import time
from collections import defaultdict
from itertools import chain

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import models
import nli_data
from nli_model import NLINet

import logger as logger_module

parser = argparse.ArgumentParser(description='NLI taining.')
parser.add_argument('--data', type=str, default='/pio/data/data/bytecnn/sentence_encoding_data/SNLI',
                    help='path to NLI data')
parser.add_argument('--glove', type=str, default='/pio/data/data/bytecnn/sentence_encoding_data/GloVe/glove.840B.300d.txt',
                    help='path to GloVe')
parser.add_argument('--data-kwargs', type=str, default='',
                    help='')
parser.add_argument('--encoder', type=str, default='WordCNNEncoder',
                    help='encoder class')
parser.add_argument('--encoder-kwargs', type=str, default='',
                    help='encoder kwargs')
parser.add_argument('--model-kwargs', type=str, default='',
                    help='model kwargs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--lr-lambda', type=str, default='lambda epoch: 0.5 ** (epoch // 10)',
                    help='learning rate based on base lr and iteration')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval-batch-size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--optimizer', default='sgd',
                    choices=('sgd', 'adam', 'adagrad', 'adadelta'),
                    help='optimization method')
parser.add_argument('--optimizer-kwargs', type=str, default='momentum=0.9,weight_decay=0.00001',
                    help='kwargs for the optimizer (e.g., momentum=0.9)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save-state', type=bool, default=True,
                    help='save training state after each epoch')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--logdir', type=str,  default=None,
                    help='path to save the final model')
parser.add_argument('--log-weights', action='store_true',
                    help="log weights' histograms")
parser.add_argument('--log-grads', action='store_true',
                    help="log gradients' histograms")
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
args = parser.parse_args()
print(args)
print()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably "
              "run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

data_kwargs = eval('dict(%s)' % args.data_kwargs)
dataset = nli_data.NLIWordCorpus(
        data_path=args.data,
        glove_path=args.glove,
        cuda=args.cuda,
        **data_kwargs)

###############################################################################
# Build the model
###############################################################################

# Evaluate this early to know which data options to use
encoder_class = getattr(models, args.encoder)
args.encoder_kwargs = args.encoder_kwargs.replace("norm=batch", "norm='batch'")

encoder_kwargs = eval("dict(%s)" % (args.encoder_kwargs,))
encoder = encoder_class(**encoder_kwargs)

model_kwargs = eval("dict(%s)" % (args.model_kwargs,))
model = NLINet(encoder=encoder, **model_kwargs)

if args.cuda:
    model.cuda()

logger_module.print_model_summary(model)

###############################################################################
# Setup training
###############################################################################

optimizer_proto = {'sgd': optim.SGD, 'adam': optim.Adam,
                   'adagrad': optim.Adagrad, 'adadelta': optim.Adadelta}
optimizer_kwargs = eval("dict(%s)" % args.optimizer_kwargs)
optimizer_kwargs['lr'] = args.lr
optimizer = optimizer_proto[args.optimizer](
    model.parameters(), **optimizer_kwargs)

if args.lr_lambda:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=eval(args.lr_lambda))
else:
    scheduler = None

if False: #args.resume_training != '': XXX
    innards = logger_module.resume_training_innards(state, model, optimizer, scheduler)
    model = innards['model']
    optimizer = innards['optimizer']
    scheduler = innards['scheduler']
    logger = innards['logger']
    first_epoch = innards['first_epoch']
else:
    logger = logger_module.Logger(
        optimizer.param_groups[0]['lr'], args.log_interval,
        dataset.train.get_num_batches(args.batch_size), logdir=args.logdir,
        log_weights=args.log_weights, log_grads=args.log_grads)
    #logger.save_model_info(dict(model=(args.model, model_kwargs))) TODO
    first_epoch = 1

print(logger.logdir)

###############################################################################
# Training code
###############################################################################


# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(first_epoch, args.epochs+1):
        logger.mark_epoch_start(epoch)

        model.train_on(dataset.train.iter_epoch(args.batch_size),
                       optimizer,
                       logger,
                       clip=args.clip)

        val_loss = model.eval_on(
                dataset.valid.iter_epoch(args.batch_size, evaluation=True))

        logger.valid_log(val_loss, mode='valid')

        if args.save_state:
            logger.save_model_state_dict(model.state_dict(), current=True)
            logger.save_training_state(
                optimizer, args,
                model_state=(model.get_state() if hasattr(model, 'get_state') else None))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
