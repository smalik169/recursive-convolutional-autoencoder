from __future__ import print_function

import codecs
import pprint
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


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
    def __init__(self, n, emsize, batch_norm, batch_norm_eval_updates=False,
            padding_idx=None):
        super(ByteCNNEncoder, self).__init__()
        self.n = n
        self.emsize = emsize
        assert n % 2 == 0, 'n should be a multiple of 2'
        conv_kwargs = dict(kernel_size=3, stride=1, padding=1, bias=False)
        conv_proto = lambda: nn.Conv1d(emsize, emsize, **conv_kwargs)
        linear_proto = lambda: nn.Linear(emsize*4, emsize*4)
        residual_list = lambda proto, k: [Residual(proto, batch_norm=batch_norm) \
                                          for _ in xrange(k)]

        self.embedding = nn.Embedding(emsize, emsize, padding_idx=padding_idx)
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
    def __init__(self, n=8, emsize=256, batch_norm=True, ignore_index=-1, eos=0):  ## XXX Check default emsize
        super(ByteCNN, self).__init__()
        self.n = n
        self.emsize = emsize
        self.batch_norm = batch_norm
        self.encoder = ByteCNNEncoder(n, emsize, batch_norm=batch_norm,
                padding_idx=(ignore_index if ignore_index >= 0 else None))
        self.decoder = ByteCNNDecoder(n, emsize, batch_norm=batch_norm)
        self.log_softmax = nn.LogSoftmax()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.eos = eos

    def forward(self, x):
        r = self.num_recurrences(x)
        x = self.encoder(x, r)
        x = self.decoder(x, r)
        return self.log_softmax(x)

    def num_recurrences(self, x):
        rfloat = np.log2(x.size(-1))
        r = int(rfloat)
        assert float(r) == rfloat
        return r

    def _encode_decode(self, src, tgt):
        r_src = self.num_recurrences(src)
        r_tgt = self.num_recurrences(tgt)
        features = self.encoder(src, r_src)
        return self.decoder(features, r_tgt)

    def train_on(self, batch_iterator, optimizer, logger=None):
        self.train()
        losses = []
        errs = []
        for batch, (src, tgt) in enumerate(batch_iterator):
            self.zero_grad()
            src = Variable(src)
            tgt = Variable(tgt)
            decoded = self._encode_decode(src, tgt)
            loss = self.criterion(
                decoded.transpose(1, 2).contiguous().view(-1, decoded.size(1)),
                tgt.view(-1))
            loss.backward()
            optimizer.step()

            _, predictions = decoded.data.max(dim=1)
            mask = (tgt.data != self.criterion.ignore_index)
            err_rate = 100. * (predictions[mask] != tgt.data[mask]).sum() / mask.sum()
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
        for (src, tgt) in batch_iterator:
            src = Variable(src, volatile=True)
            tgt = Variable(tgt, volatile=True)
            decoded = self._encode_decode(src, tgt)
            total_loss += self.criterion(
                decoded.transpose(1, 2).contiguous().view(-1, decoded.size(1)),
                tgt.view(-1))

            _, predictions = decoded.data.max(dim=1)
            mask = (tgt.data != self.criterion.ignore_index)
            errs += (predictions[mask] != tgt.data[mask]).sum()
            samples += mask.sum()
            batch_cnt += 1
        return {'loss': total_loss.data[0]/batch_cnt,
                'acc': 100 - 100. * errs / samples,}

    def try_on(self, batch_iterator, switch_to_evalmode=True):
        self.eval() if switch_to_evalmode else self.train()
        predicted = []
        for (src, tgt) in batch_iterator:
            src = Variable(src, volatile=True)
            tgt = Variable(tgt, volatile=True)
            decoded = self._encode_decode(src, tgt)
            _, predictions = decoded.data.max(dim=1)

            # Make into strings and append to decoded
            for pred in predictions:
                pred = list(pred.cpu().numpy())
                pred = pred[:pred.index(self.eos)] if self.eos in pred else pred
                pred = repr(''.join([chr(c) for c in pred]))
                predicted.append(pred)
        return predicted

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


class VAEByteCNN(nn.Module):
    save_best = True
    def __init__(self, n=8, emsize=256, batch_norm=True,  ## XXX Check default emsize
                 ignore_index=-1, eos=0,
                 kl_weight_init=1e-5, kl_weight_end=1.0,
                 kl_increment_start=None, kl_increment=None):
        super(VAEByteCNN, self).__init__()
        self.n = n
        self.emsize = emsize
        self.batch_norm = batch_norm
        self.encoder = ByteCNNEncoder(n, emsize, batch_norm=batch_norm,
                padding_idx=(ignore_index if ignore_index >= 0 else None))
        self.decoder = ByteCNNDecoder(n, emsize, batch_norm=batch_norm)
        self.log_softmax = nn.LogSoftmax()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.eos = eos
        self.projection = nn.Linear(4*emsize, 8*emsize)

        self.kl_weight = kl_weight_init
        self.kl_weight_end = kl_weight_end
        self.kl_increment_start = kl_increment_start
        self.kl_increment = kl_increment

    def get_features_and_KL(self, mu, log_sigma):
        bs = mu.size(0)
        dim = mu.size(1)
        sigma = torch.exp(log_sigma)
        kl = -0.5 * torch.sum((1.0 + 2.0 * log_sigma - mu**2 - sigma**2) / bs)
        epsilon = mu.data.new(bs, dim).normal_()
        epsilon = Variable(epsilon)
        features = epsilon * sigma + mu
        return features, kl

    def forward(self, x):
        r = self.encoder.num_recurrences(x)
        x = self.encoder(x, r)
        x = self.decoder(x, r)
        return self.log_softmax(x)

    def num_recurrences(self, x):
        rfloat = np.log2(x.size(-1))

        r = int(rfloat)
        assert float(r) == rfloat
        return r

    def _encode_decode(self, src, tgt):
        r_src = self.num_recurrences(src)
        r_tgt = self.num_recurrences(tgt)
        dist_params = self.projection(self.encoder(src, r_src))
        mu, log_sigma = dist_params.chunk(2, dim=1)
        log_sigma /= 33.0
        features, kl = self.get_features_and_KL(mu, log_sigma)
        return self.decoder(features, r_tgt), kl

    def train_on(self, batch_iterator, optimizer, logger=None):
        self.train()
        losses = []
        errs = []
        for batch, (src, tgt) in enumerate(batch_iterator):
            self.zero_grad()
            src = Variable(src)
            tgt = Variable(tgt)
            decoded, kl = self._encode_decode(src, tgt)
            loss = self.criterion(
                decoded.transpose(1, 2).contiguous().view(-1, decoded.size(1)),
                tgt.view(-1)) + self.kl_weight * kl
            loss.backward()
            optimizer.step()

            _, predictions = decoded.data.max(dim=1)
            mask = (tgt.data != self.criterion.ignore_index)
            err_rate = 100. * (predictions[mask] != tgt.data[mask]).sum() / mask.sum()
            losses.append(loss.data[0])
            errs.append(err_rate)
            logger.train_log(batch, {'loss': loss.data[0], 'acc': 100. - err_rate,
                                     'kl': kl.data[0], 'kl_weight': self.kl_weight},
                             named_params=self.named_parameters)

            if self.kl_increment_start > 0:
                self.kl_increment_start -= 1
            else:
                self.kl_weight = min(self.kl_weight_end,
                                     self.kl_weight + self.kl_increment)
        return losses, errs

    def eval_on(self, batch_iterator, switch_to_evalmode=True):
        self.eval() if switch_to_evalmode else self.train()
        errs = 0
        samples = 0
        total_loss = 0
        batch_cnt = 0
        for batch, (src, tgt) in enumerate(batch_iterator):
            src = Variable(src, volatile=True)
            tgt = Variable(tgt, volatile=True)
            decoded, kl = self._encode_decode(src, tgt)
            total_loss += self.criterion(
                decoded.transpose(1, 2).contiguous().view(-1, decoded.size(1)),
                tgt.view(-1)) + self.kl_weight * kl

            _, predictions = decoded.data.max(dim=1)
            mask = (tgt.data != self.criterion.ignore_index)
            errs += (predictions[mask] != tgt.data[mask]).sum()
            samples += mask.sum()
            batch_cnt += 1
        return {'loss': total_loss.data[0]/batch_cnt,
                'acc': 100 - 100. * errs / samples,
                'kl': kl.data[0], 'kl_weight': self.kl_weight}

    def try_on(self, batch_iterator, switch_to_evalmode=True):
        self.eval() if switch_to_evalmode else self.train()
        predicted = []
        for batch, (src, tgt) in enumerate(batch_iterator):
            src = Variable(src, volatile=True)
            tgt = Variable(tgt, volatile=True)
            decoded, kl = self._encode_decode(src, tgt)
            _, predictions = decoded.data.max(dim=1)

            # Make into strings and append to decoded
            for pred in predictions:
                pred = list(pred.cpu().numpy())
                pred = pred[:pred.index(self.eos)] if self.eos in pred else pred
                pred = repr(''.join([chr(c) for c in pred]))
                predicted.append(pred)
        return predicted

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

        assert p['model_class'] == 'VAEByteCNN', \
            'Tried to load %s as ByteCNN' % p['model_class']
        model = model_class(**model_kwargs)
        with open(model_pt, 'rb') as f:
            model.load_state_dict(torch.load(f))
        return model
