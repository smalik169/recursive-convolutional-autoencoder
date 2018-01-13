#!/usr/bin/env python
from __future__ import print_function

import argparse
import codecs
import pprint
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import model
from logger import Logger


class UTF8File(object):

    def __init__(self, path):
        self.lines = [[ord(c) for c in l.strip().encode('utf-8')] \
                      for l in codecs.open(path, 'r', 'utf-8')]
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)

    def get_num_batches(self, bsz):
        return self.lines // bsz

    def iter_epoch(self, bsz, evaluation=False):
        return NotImplementedError


class UTF8Corpus(object):
    def __init__(self, path, cuda, rng=None):
        self.train = UTF8File(path + 'train.txt', cuda, rng=rng)
        self.valid = UTF8File(path + 'test.txt', cuda, rng=rng)
        self.test = UTF8File(path + 'valid.txt', cuda, rng=rng)


class ExpandConv1d(nn.Module):
    def forward(self, x):
        # Output of conv1d: (N,Cout,Lout)
        bsz, c, l = x.size()
        x = x.view(bsz, c // 2, 2, l).transpose(2, 3).contiguous()
        return x.view(bsz, c // 2, 2 * l).contiguous()


class ByteCNNEncoder(nn.Module):
    save_best = True
    def __init__(self, n, emsize):
        super(ByteCNNEncoder, self).__init__()

        self.n = n

        # Input: LongTensor (N, W), N = mini-batch, W = number of indices to extract per mini-batch
        # Output: (N, W, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, emsize)

        conv_block = lambda n: [nn.Conv1d(256, 256, 3, padding=1) \
                                for _ in xrange(n)]

        self.prefix = nn.Sequential(*conv_block(n))
        self.recurrent = nn.Sequential(*conv_block(n))
        self.recurrent.add_module(nn.Pooling())
        # TODO Reshape, add ReLUs

        linear_block = lambda n: [nn.Conv1d(256, 256, 3, padding=1) \
                                  for _ in xrange(n)]

        self.postfix = nn.Sequential(nn.Linear() for _ in range(n))

    def forward(self, x):
        x = self.embedding(x).t(1, 2)
        x = self.prefix(x)

        rfloat = np.log(x.size(-1))
        r = int(r)
        assert float(r) == rfloat

        for _ in xrange(r):
            x = self.recurrent(x)

        assert x.size(-1) == 1024 # XXX
        return self.postfix(x)


class ByteCNNDecoder(nn.Module):
    save_best = True
    def __init__(self, n, emsize):
        super(ByteCNNDecoder, self).__init__()

        self.n = n
        self.embedding = nn.Embedding(num_embeddings, emsize)

        self.prefix = nn.ModuleList([nn.Linear() for _ in xrange(n)])
        self.recurrent = nn.ModuleList([nn.Conv1d() for _ in xrange(n)] + [nn.ExpandConv()])
        self.postfix = nn.ModuleList([nn.Conv1d() for _ in xrange(n)])


class ByteCNN(nn.module):
    def __init__(self, ):
        self.encoder = ByteCNNEncoder()
        self.decoder = ByteCNNDecoder()


parser = argparse.ArgumentParser(description='Byte-level CNN text autoencoder.')
parser.add_argument('--resume-training', type=str, default='',
                    help='path to a training directory (loads the model and the optimizer)')
# TODO Change to --resume-force-args, which passes dict of input args to be overwritten (e.g., "dict(epochs=42)")
# parser.add_argument('--resume-epochs', type=int, default=None,
#                     help='force a number of epochs when loading an experiment')
parser.add_argument('--data', type=str, default='TODO',
                    help='name of the dataset')
parser.add_argument('--model', type=str, default='ByteCNN',
                    help='model class')
parser.add_argument('--model-kwargs', type=str, default='',
                    help='model kwargs')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--lr-lambda', type=str, default=None, # 'lambda lr,it: lr',
                    help='learning rate based on base lr and iteration')
# parser.add_argument('--clip', type=float, default=0.25,
#                     help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--optimizer', default='sgd',
                    choices=('sgd', 'adam', 'adagrad', 'adadelta'),
                    help='optimization method')
parser.add_argument('--optimizer-kwargs', type=str, default='',
                    help='kwargs for the optimizer (e.g., momentum=0.9)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save-state', action='store_true',
                    help='save training state after each epoch')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--logdir', type=str,  default=None,
                    help='path to save the final model')
# parser.add_argument('--save', type=str,  default='model.pt',
#                     help='path to save the final model')
parser.add_argument('--log-weights', action='store_true',
                    help="log weights' histograms")
parser.add_argument('--log-grads', action='store_true',
                    help="log gradients' histograms")
args = parser.parse_args()

###############################################################################
# Resume old training?
###############################################################################

if args.resume_training != '':
    # Overwrite the args with loaded ones, build the model, optimizer, corpus
    # This will allow to keep things similar, e.g., initialize corpus with
    # a proper random seed (which will later get overwritten)
    resume_path = args.resume_training
    print('\nResuming training of %s' % resume_path)
    print('\nWarning: Ignoring other input arguments!\n')
    state = Logger.load_training_state(resume_path)
    state['args'].__dict__['resume_training'] = resume_path # XXX
    if args.resume_epochs is not None:
        state['args'].__dict__['epochs'] = args.resume_epochs
    args = state['args']

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

dataset = UTF8Corpus(args.data, batch_size=args.batch_size, cuda=args.cuda)

###############################################################################
# Build the model
###############################################################################

# Evaluate this early to know which data options to use
model_kwargs = eval("dict(%s)" % (args.model_kwargs,))
model = ByteCNN(**model_kwargs)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in model_parameters])
print("Model summary:\n%s" % (model,))
print("Model params:\n%s" % ("\n".join(
    ["%s: %s" % (p[0], p[1].size()) for p in model.named_parameters()])))
print("Number of params: %d" % num_params)

###############################################################################
# Training code
###############################################################################

if args.cuda:
    model.cuda()

optimizer_proto = {'sgd': optim.SGD, 'adam': optim.Adam,
                   'adagrad': optim.Adagrad, 'adadelta': optim.Adadelta}
optimizer_kwargs = eval("dict(%s)" % args.optimizer_kwargs)
optimizer_kwargs['lr'] = args.lr
optimizer = optimizer_proto[args.optimizer](
    model.parameters(), **optimizer_kwargs)

if args.lr_lambda is not None:
    lr_decay = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=eval(args.lr_lambda))
else:
    lr_decay = None

if args.resume_training != '':
    # State has been loaded before model construction
    logger = state['logger']
    state = logger.set_training_state(state, optimizer)
    optimizer = state['optimizer']
    model.load_state_dict(logger.load_model_state_dict(current=True))
    first_epoch = logger.epoch + 1
else:
    logger = None
    first_epoch = 1

# Loop over epochs.
# XXX Save it in training state?
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    if logger is None:
        logger = Logger(optimizer.param_groups[0]['lr'], args.log_interval,
                        dataset.get_num_batches(args.batch_size), logdir=args.logdir,
                        log_weights=args.log_weights, log_grads=args.log_grads)
        # XXX
        # logger.save_model_info(args.model, generator_kwargs,
        #         args.initializer_class, initializer_kwargs)
    print(logger.logdir)

    for epoch in range(first_epoch, args.epochs+1):
        logger.mark_epoch_start(epoch)

        model.train_on(dataset['train'], optimizer, logger)
        val_loss = model.eval_on(dataset['valid'])
        logger.valid_log(val_loss)

        # Save the model if the validation loss is the best we've seen so far.
        if args.save_state:
            logger.save_model_state_dict(model.state_dict(), current=True)
            logger.save_training_state(optimizer, args)

        if model.save_best and False: # not best_val_loss or val_loss['nll_per_w'] < best_val_loss:
                logger.save_model_state_dict(model.state_dict())
                #logger.save_model(model)
                best_val_loss = val_loss['nll_per_w']

        if lr_decay is not None:
            lr_decay.step()
            # XXX print (if not logging already?) optimizer.param_groups[0]['lr']

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

sys.exit(0) # XXX

# Load the best saved model.
# model = logger.load_model()
model.load_state_dict(logger.load_model_state_dict())

# Run on all data
# train_loss = model.eval_on(
#     corpus.train.iter_epoch(eval_batch_size, args.bptt, evaluation=True))
# valid_loss = model.eval_on(
#     corpus.valid.iter_epoch(eval_batch_size, args.bptt, evaluation=True))
# results = dict(train=train_loss, valid=valid_loss, test=test_loss)

test_loss = model.eval_on(
    corpus.test.iter_epoch(eval_batch_size, args.bptt, evaluation=True))
results = dict(test=test_loss)

logger.final_log(results)

# Run on test data.
corpus.valid.iter_epoch(eval_batch_size, args.bptt, evaluation=True)
test_loss = model.eval_on(
    corpus.test.iter_epoch(eval_batch_size, args.bptt, evaluation=True))
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

model_logger.save_results(results, model_path=args.save)

# def logging_callback(batch, batch_loss):
#     global total_loss
#     global minibatch_start_time
#     total_loss += batch_loss
#     if batch % args.log_interval == 0 and batch > 0:
#         cur_loss = total_loss[0] / args.log_interval
#         elapsed = (time.time() - minibatch_start_time
#                    ) * 1000 / args.log_interval
#         print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | '
#               'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
#                 epoch, batch, num_batches, optimizer.param_groups[0]['lr'],
#                 elapsed, cur_loss, math.exp(cur_loss)))
#         total_loss = 0
#         minibatch_start_time = time.time()

