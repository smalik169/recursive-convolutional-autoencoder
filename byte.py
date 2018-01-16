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

from logger import Logger

parser = argparse.ArgumentParser(description='Byte-level CNN text autoencoder.')
parser.add_argument('--resume-training', type=str, default='',
                    help='path to a training directory (loads the model and the optimizer)')
parser.add_argument('--resume-training-force-args', type=str, default='',
                    help='list of input args to be overwritten when resuming (e.g., # of epochs)')
parser.add_argument('--data', type=str, default='TODO',
                    help='name of the dataset')
parser.add_argument('--model', type=str, default='ByteCNN',
                    help='model class')
parser.add_argument('--model-kwargs', type=str, default='',
                    help='model kwargs')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
# Default from the Byte-level CNN paper: half lr every 10 epochs
parser.add_argument('--lr-lambda', type=str, default='lambda epoch: 0.5 ** (epoch // 10)',
                    help='learning rate based on base lr and iteration')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval-batch-size', type=int, default=10, metavar='N',
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


class UTF8File(object):
    EOS = 256  # XXX XXX XXX additional non-a-byte symbol
    def __init__(self, path, cuda, rng=None):
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)

        lines_by_len = defaultdict(list)
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                bytes_ = [ord(c) for c in line.strip()] + [self.EOS]
                bytes_ += [self.EOS] * (int(2 ** np.ceil(np.log2(len(bytes_)))) - len(bytes_))
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
                    yield batch_tensor.cuda() if self.cuda else batch_tensor
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
                yield batch_tensor.cuda() if self.cuda else batch_tensor


class UTF8Corpus(object):
    def __init__(self, path, cuda, rng=None):
        self.train = UTF8File(path + 'train.txt', cuda, rng=rng)
        self.valid = UTF8File(path + 'test.txt', cuda, rng=rng)
        self.test = UTF8File(path + 'valid.txt', cuda, rng=rng)


class ExpandConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ExpandConv1d, self).__init__()
        self.conv1d = nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        # Output of conv1d: (N,Cout,Lout)
        x = self.conv1d(x)
        bsz, c, l = x.size()
        x = x.view(bsz, c // 2, 2, l).transpose(2, 3).contiguous()
        return x.view(bsz, c // 2, 2 * l).contiguous()


def insert_relu(layer_list, last=True):
    ret = list(chain.from_iterable(zip(layer_list,
                                       [nn.ReLU() for _ in layer_list])))
    return ret if last else ret[:-1]


class ByteCNNEncoder(nn.Module):
    def __init__(self, n, emsize=257):
        super(ByteCNNEncoder, self).__init__()
        conv_block = lambda i: [nn.Conv1d(emsize, emsize, 3, padding=1) \
                                for _ in xrange(i)]
        linear_block = [nn.Linear(emsize * 4, emsize * 4) for _ in xrange(n)]

        self.n = n
        self.embedding = nn.Embedding(emsize, emsize)
        self.prefix = nn.Sequential(*insert_relu(conv_block(n), last=True))
        self.recurrent = nn.Sequential(*insert_relu(conv_block(n), last=True))
        self.recurrent.add_module(module=nn.MaxPool1d(kernel_size=2),
                                  name='max_pool')
        self.postfix = nn.Sequential(*insert_relu(linear_block, last=False))

    def forward(self, x, r):
        x = self.embedding(x).transpose(1, 2)
        x = self.prefix(x)

        for _ in xrange(r-2):
            x = self.recurrent(x)

        bsz = x.size(0)
        return self.postfix(x.view(bsz, -1))

    def num_recurrences(self, x):
        rfloat = np.log2(x.size(-1))
        r = int(rfloat)
        assert float(r) == rfloat
        return r


class ByteCNNDecoder(nn.Module):
    def __init__(self, n, emsize):
        super(ByteCNNDecoder, self).__init__()
        conv_block_fun = lambda i: [nn.Conv1d(emsize, emsize, 3, padding=1) \
                                    for _ in xrange(i)]
        linear_block = [(nn.Linear(emsize * 4, emsize * 4), nn.ReLU())]
        linear_block = [l for tupl in linear_block for l in tupl]
        linear_block.append(nn.Linear(emsize * 4, emsize * 4))

        self.n = n
        self.emsize = emsize
        self.prefix = nn.Sequential(*linear_block)
        self.recurrent = nn.Sequential(*([ExpandConv1d(emsize, emsize * 2, 3, padding=1)] +\
                                         conv_block_fun(n)))
        self.postfix = nn.Sequential(*conv_block_fun(n))

    def forward(self, x, r):
        x = self.prefix(x)
        x = x.view(x.size(0), self.emsize, 4)

        for _ in xrange(r-2):
            x = self.recurrent(x)
        return self.postfix(x)


class ByteCNN(nn.Module):
    save_best = True
    def __init__(self, n=8, emsize=256):
        super(ByteCNN, self).__init__()
        self.n = n
        self.emsize = emsize
        self.encoder = ByteCNNEncoder(n, emsize)
        self.decoder = ByteCNNDecoder(n, emsize)
        self.log_softmax = nn.LogSoftmax()
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        r = self.encoder.num_recurrences(x)
        x = self.encoder(x, r)
        x = self.decoder(x, r-1)
        return self.log_softmax(x)

    def train_on(self, batch_iterator, optimizer, logger=None):
        self.train()
        losses = []
        errs = []
        for batch, src in enumerate(batch_iterator):
            self.zero_grad()
            src = Variable(src)
            r = self.encoder.num_recurrences(src)
            features = self.encoder(src, r)
            tgt = self.decoder(features, r)
            loss = self.criterion(
                tgt.transpose(1, 2).contiguous().view(-1, tgt.size(1)),
                src.view(-1))
            loss.backward()
            optimizer.step()

            _, predictions = tgt.data.max(dim=1)
            err_rate = 100. * (predictions != src.data).sum() / np.prod(src.size())
            losses.append(loss.data[0])
            errs.append(err_rate)
            logger.train_log(batch, {'loss': loss.data[0], 'acc': 100. - err_rate,}, #loss.data[0]},
                             named_params=self.named_parameters)
        return losses, errs

    def eval_on(self, batch_iterator):
        self.eval()
        errs = 0
        samples = 0
        total_loss = 0
        for src in batch_iterator:
            src = Variable(src, volatile=True)
            r = self.encoder.num_recurrences(src)
            features = self.encoder(src, r)
            tgt = self.decoder(features, r)
            total_loss += self.criterion(
                tgt.transpose(1, 2).contiguous().view(-1, tgt.size(1)),
                src.view(-1))

            _, predictions = tgt.data.max(dim=1)
            errs += (predictions != src.data).sum()
            samples += np.prod(src.size())
        return {'loss': total_loss.data[0], 'acc': 100 - 100. * errs / samples}

    @staticmethod
    def load_model(path):
        """Load a model"""
        model_pt = os.path.join(path, 'model.pt')
        model_info = os.path.join(path, 'model.info')
    
        with open(model_info, 'r') as f:
            p = defaultdict(str)
            p.update(dict(line.strip().split('=', 1) for line in f))
    
        # Read and pop one by one, then raise if something's left
        model_class = eval(p['model_class'])
        del p['model_class']
        model_kwargs = eval("dict(%s)" % p['model_kwargs'])
        del p['model_kwargs']
        if len(p) > 0:
            raise ValueError('Unknown model params: ' + ', '.join(p.keys()))
       
        assert p['model_class'] == 'ByteCNN', \
            'Tried to load %s as ByteCNN' % p['model_class'] 
        model = model_class(**model_kwargs)
        with open(model_pt, 'rb') as f:
            model.load_state_dict(torch.load(f))
        return model


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

    if args.resume_training_force_args != '':
        print('\nForcing args: %s' % args.resume_training_force_args)
        forced_args = eval('dict(%s)' % args.resume_training_force_args)
        args.update(forced_args)

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

dataset = UTF8Corpus(args.data, cuda=args.cuda)

###############################################################################
# Build the model
###############################################################################

# Evaluate this early to know which data options to use
model_kwargs = eval("dict(%s)" % (args.model_kwargs,))
model = ByteCNN(**model_kwargs)

if args.cuda:
    model.cuda()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in model_parameters])
print("Model summary:\n%s" % (model,))
print("Model params:\n%s" % ("\n".join(
    ["%s: %s" % (p[0], p[1].size()) for p in model.named_parameters()])))
print("Number of params: %.2fM" % (num_params / 10.0**6))

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
    logger = Logger(optimizer.param_groups[0]['lr'], args.log_interval,
                    dataset.train.get_num_batches(args.batch_size), logdir=args.logdir,
                    log_weights=args.log_weights, log_grads=args.log_grads)
    logger.save_model_info(dict(model=(args.model, model_kwargs)))
    first_epoch = 1
print(logger.logdir)

###############################################################################
# Training code
###############################################################################
logger.save_model_state_dict(model.state_dict())

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(first_epoch, args.epochs+1):
        logger.mark_epoch_start(epoch)

        model.train_on(dataset.train.iter_epoch(args.batch_size),
                       optimizer, logger)
        val_loss = model.eval_on(dataset.valid.iter_epoch(args.batch_size,
                                                          evaluation=True))
        logger.valid_log(val_loss)

        # Save the model if the validation loss is the best we've seen so far.
        if args.save_state:
            logger.save_model_state_dict(model.state_dict(), current=True)
            logger.save_training_state(optimizer, args)

        if model.save_best and False: # not best_val_loss or val_loss['nll_per_w'] < best_val_loss:
                logger.save_model_state_dict(model.state_dict())
                best_val_loss = val_loss['nll_per_w']

        # TODO Save training state

        if lr_decay is not None:
            lr_decay.step()



except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


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
    dataset.test.iter_epoch(args.eval_batch_size, evaluation=True))
results = dict(test=test_loss)

logger.final_log(results)

# Run on test data.
#corpus.valid.iter_epoch(eval_batch_size, args.bptt, evaluation=True)
#test_loss = model.eval_on(
#    dataset.test.iter_epoch(eval_batch_size, args.bptt, evaluation=True))
#print('=' * 89)
#print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#    test_loss, math.exp(test_loss)))
#print('=' * 89)

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

