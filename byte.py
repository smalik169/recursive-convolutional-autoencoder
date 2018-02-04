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
parser.add_argument('--data', type=str, default='./data/ptb.',
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
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval-batch-size', type=int, default=10, metavar='N',
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
        self.lines = {k: np.asarray(v, dtype=np.int32) \
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

    def sample_batch(self, bsz, sample_sentence=None): 
        if not sample_sentence:
            sample_sentence = 'On a beautiful morning, a busty Amazon rode through a forest.' 
        sample_sentence = sample_sentence.encode('utf-8')
        batch_len = int(2 ** np.ceil(np.log2(len(sample_sentence) + 1)))
        bytes_ = np.asarray([[ord(c) for c in sample_sentence] + [self.EOS] + \
                             [self.EMPTY] * (batch_len - len(sample_sentence) - 1)],
                            dtype=np.int32) 
        assert bytes_.shape[1] == batch_len, bytes_.shape 
        # batch_tensor = torch.from_numpy(bytes_).long() 
        inds = np.random.choice(len(self.lines[batch_len]), bsz) 
        batch_tensor = torch.from_numpy(self.lines[batch_len][inds]).long() 
        batch_tensor[0] = torch.from_numpy(bytes_).long() 
        yield batch_tensor.cuda() if self.cuda else batch_tensor 


class UTF8Corpus(object):
    def __init__(self, path, cuda, rng=None):
        self.train = UTF8File(path + 'train.txt', cuda, rng=rng)
        self.valid = UTF8File(path + 'valid.txt', cuda, rng=rng)
        self.test = UTF8File(path + 'test.txt', cuda, rng=rng)


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


class Residual(nn.Module):
    def __init__(self, layer_proto, layer2_proto=None, out_relu=True,
                 batch_norm=True):
        super(Residual, self).__init__()
        self.layer1 = layer_proto()
        self.relu = nn.ReLU()
        self.layer2 = layer2_proto() if layer2_proto else layer_proto()
        self.out_relu = out_relu
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(self.num_channels(self.layer1), momentum=0.1)
            self.bn2 = nn.BatchNorm1d(self.num_channels(self.layer2), momentum=0.1)

    def num_channels(self, layer):
        if type(layer) is ExpandConv1d:
            return layer.conv1d.out_channels
        elif type(layer) is nn.Conv1d:
            return layer.out_channels
        elif type(layer) is nn.Linear:
            return layer.weight.size(0)
        else:
            raise ValueError('Unsupported layer type.')

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.layer2(out)
        if self.batch_norm:
            out = self.bn2(out)

        out += residual
        if self.out_relu:
            out = self.relu(out)
        return out


class ByteCNNEncoder(nn.Module):
    def __init__(self, n, emsize, batch_norm, batch_norm_eval_updates=False):
        super(ByteCNNEncoder, self).__init__()
        self.n = n
        self.emsize = emsize
        assert n % 2 == 0, 'n should be a multiple of 2'
        conv_kwargs = dict(kernel_size=3, stride=1, padding=1, bias=False)
        conv_proto = lambda: nn.Conv1d(emsize, emsize, **conv_kwargs)
        linear_proto = lambda: nn.Linear(emsize*4, emsize*4)
        residual_list = lambda proto, k: [Residual(proto, batch_norm=batch_norm) \
                                          for _ in xrange(k)]

        self.embedding = nn.Embedding(emsize, emsize, padding_idx=UTF8File.EMPTY)
        self.prefix = nn.Sequential(*(residual_list(conv_proto, n//2)))
        self.recurrent = nn.Sequential(*(residual_list(conv_proto, n//2) + \
                                         [nn.MaxPool1d(kernel_size=2)]))
        self.postfix = nn.Sequential(*(residual_list(linear_proto, n//2-1) + \
                                       [Residual(linear_proto, out_relu=False, batch_norm=batch_norm)]))

        self.batch_norm = batch_norm
        self.batch_norm_eval_updates = batch_norm_eval_updates

    def forward(self, x, r):
        assert x.size(1) >= 4
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
    def __init__(self, n, emsize, batch_norm):
        super(ByteCNNDecoder, self).__init__()
        self.n = n
        self.emsize = emsize
        assert n % 2 == 0, 'n should be a multiple of 2'
        conv_kwargs = dict(kernel_size=3, stride=1, padding=1, bias=False)
        conv_proto = lambda: nn.Conv1d(emsize, emsize, **conv_kwargs)
        expand_proto = lambda: ExpandConv1d(emsize, emsize*2, **conv_kwargs)
        linear_proto = lambda: nn.Linear(emsize*4, emsize*4)
        bn_proto = lambda c: nn.BatchNorm1d(c, momentum=0.1)
        residual_list = lambda proto, k: [Residual(proto, batch_norm=batch_norm) \
                                          for _ in xrange(k)]

        self.prefix = nn.Sequential(*(residual_list(linear_proto, n//2)))
        if batch_norm:
            self.recurrent = nn.Sequential(
                *([expand_proto(), bn_proto(emsize), nn.ReLU(), conv_proto(),
                   bn_proto(emsize), nn.ReLU()] + \
                  residual_list(conv_proto, n//2-1)))
        else:
            self.recurrent = nn.Sequential(
                *([expand_proto(), nn.ReLU(), conv_proto(), nn.ReLU()] + \
                  residual_list(conv_proto, n//2-1)))
        self.postfix = nn.Sequential(*(residual_list(conv_proto, n//2)))

    def forward(self, x, r):
        x = self.prefix(x)
        x = x.view(x.size(0), self.emsize, 4)

        for _ in xrange(r-2):
            x = self.recurrent(x)

        return self.postfix(x)


class ByteCNN(nn.Module):
    save_best = True
    def __init__(self, n=8, emsize=256, batch_norm=True):  ## XXX Check default emsize
        super(ByteCNN, self).__init__()
        self.n = n
        self.emsize = emsize
        self.batch_norm = batch_norm
        self.encoder = ByteCNNEncoder(n, emsize, batch_norm=batch_norm)
        self.decoder = ByteCNNDecoder(n, emsize, batch_norm=batch_norm)
        self.log_softmax = nn.LogSoftmax()
        #self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss(ignore_index=UTF8File.EMPTY)

    def forward(self, x):
        r = self.encoder.num_recurrences(x)
        x = self.encoder(x, r)
        x = self.decoder(x, r)
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
            logger.train_log(batch, {'loss': loss.data[0], 'acc': 100. - err_rate,},
                             named_params=self.named_parameters)
        return losses, errs

    def eval_on(self, batch_iterator, switch_to_evalmode=True):
        self.eval() if switch_to_evalmode else self.train()
        errs = 0
        samples = 0
        total_loss = 0
        batch_cnt = 0
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
            batch_cnt += 1
        return {'loss': total_loss.data[0]/batch_cnt,
                'acc': 100 - 100. * errs / samples}

    def try_on(self, batch_iterator, switch_to_evalmode=True):
        self.eval() if switch_to_evalmode else self.train()
        decoded = []
        for src in batch_iterator:
            src = Variable(src, volatile=True)
            r = self.encoder.num_recurrences(src)
            features = self.encoder(src, r)
            tgt = self.decoder(features, r)
            _, predictions = tgt.data.max(dim=1)

            # Make into strings and append to decoded
            for pred in predictions:
                pred = list(pred.cpu().numpy())
                pred = pred[:pred.index(UTF8File.EOS)] if UTF8File.EOS in pred else pred
                pred = repr(''.join([chr(c) for c in pred]))
                decoded.append(pred)
        return decoded

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


if __name__ == '__main__':
    
    ###############################################################################
    # Resume old training?
    ###############################################################################
    
    state = None
    forced_args = None
    if args.resume_training != '':
        # Overwrite the args with loaded ones, build the model, optimizer, corpus
        # This will allow to keep things similar, e.g., initialize corpus with
        # a proper random seed (which will later get overwritten)
        args, forced_args, state = logger.resume_training(args)
    
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
        # TODO Check how it behaves on resuming training
        lr_decay = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=eval(args.lr_lambda))
    else:
        lr_decay = None
    
    if args.resume_training != '':
        # State has been loaded before model construction
        logger = state['logger']
        state = logger.set_training_state(state, optimizer)
        optimizer = state['optimizer']
    
        if forced_args and forced_args.has_key('lr'):
            optimizer.param_groups[0]['lr'] = forced_args['lr']
            logger.lr = forced_args['lr']
    
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
    
            # BatchNorm works better in train() mode (related to its running avgs)?
            # https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/3?u=smth
            val_loss = model.eval_on(
                dataset.valid.iter_epoch(args.batch_size, evaluation=True),
                switch_to_evalmode=False)
            print(model.try_on(dataset.valid.sample_batch(),
                               switch_to_evalmode=False)[0], args.batch_size)
            logger.valid_log(val_loss)
    
            # Save the model if the validation loss is the best we've seen so far.
            if args.save_state:
                logger.save_model_state_dict(model.state_dict(), current=True)
                logger.save_training_state(optimizer, args)
    
            # XXX
            # if model.save_best and False: # not best_val_loss or val_loss['nll_per_w'] < best_val_loss:
            #         logger.save_model_state_dict(model.state_dict())
            #         best_val_loss = val_loss['nll_per_w']
    
            if lr_decay is not None:
                lr_decay.step()
                logger.lr = optimizer.param_groups[0]['lr']
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
