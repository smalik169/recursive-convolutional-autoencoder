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
                 batch_norm=True, residual_connection=True):
        super(Residual, self).__init__()
        self.layer1 = layer_proto()
        self.relu = nn.ReLU()
        self.layer2 = layer2_proto() if layer2_proto else layer_proto()
        self.out_relu = out_relu
        self.batch_norm = batch_norm
        self.residual_connection = residual_connection

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

        if self.residual_connection:
            out += residual
        if self.out_relu:
            out = self.relu(out)
        return out


class ByteCNNEncoder(nn.Module):
    def __init__(self, n, emsize, batch_norm, batch_norm_eval_updates=False,
                 padding_idx=None, linear_layers=False,
                 compress_channels=None):
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
        self.linear_layers = linear_layers
        self.compress_channels = compress_channels

        if self.compress_channels:
            em_in = emsize
            postfix_layers = []
            while em_in > self.compress_channels:
                postfix_layers.append(Residual(
                    lambda: nn.Conv1d(em_in, em_in, **conv_kwargs),
                    lambda: nn.Conv1d(em_in, em_in // 2, **conv_kwargs),
                    batch_norm=batch_norm, residual_connection=False))
                em_in = em_in // 2
            self.postfix = nn.Sequential(*(postfix_layers))

        elif self.linear_layers:
            self.postfix = nn.Sequential(*(residual_list(linear_proto, n//2-1) + \
                                           [Residual(linear_proto, out_relu=False, batch_norm=batch_norm)]))
        else:
            self.postfix = None

        self.batch_norm = batch_norm
        self.batch_norm_eval_updates = batch_norm_eval_updates

    def forward(self, x, r):
        assert x.size(1) >= 4
        x = self.embedding(x).transpose(1, 2)
        x = self.prefix(x)

        for _ in xrange(r-2):
            x = self.recurrent(x)

        bsz = x.size(0)

        if self.compress_channels:
            return self.postfix(x).view(bsz, -1)
        elif self.linear_layers:
            return self.postfix(x.view(bsz, -1))
        else:
            assert self.postfix is None
            return x.view(bsz, -1)


class ByteCNNDecoder(nn.Module):
    def __init__(self, n, emsize, batch_norm, linear_layers=True,
                 compress_channels=None, output_embeddings_init=None):
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

        self.linear_layers = linear_layers
        self.compress_channels = compress_channels

        if self.compress_channels:
            em_in = self.compress_channels
            prefix_layers = []
            while em_in < self.emsize:
                prefix_layers.append(Residual(
                    lambda: nn.Conv1d(em_in, em_in, **conv_kwargs),
                    lambda: nn.Conv1d(em_in, em_in * 2, **conv_kwargs),
                    batch_norm=batch_norm, residual_connection=False))
                em_in = em_in * 2
            self.prefix = nn.Sequential(*(prefix_layers))
        elif self.linear_layers:
            self.prefix = nn.Sequential(*(residual_list(linear_proto, n//2)))
        else:
            self.prefix = None

        self.unembedding = None
        if output_embeddings_init:
            self.unembedding = nn.Linear(emsize, emsize)
            self.unembedding.weight = output_embeddings_init.weight

    def forward(self, x, r):
        if self.compress_channels:
            x = x.view(x.size(0), self.compress_channels, 4)
            x = self.prefix(x)
        elif self.linear_layers:
            x = self.prefix(x)
            x = x.view(x.size(0), self.emsize, 4)
        else:
            assert self.prefix is None
            x = x.view(x.size(0), self.emsize, 4)

        for _ in xrange(r-2):
            x = self.recurrent(x)

        x = self.postfix(x)

        if self.unembedding:
            x = self.unembedding(x.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        return x


class ByteCNN(nn.Module):
    save_best = True
    def __init__(self, n=8, emsize=256, batch_norm=True, ignore_index=-1, eos=0,
                 linear_layers=True, compress_channels=None, output_embeddings=False):  ## XXX Check default emsize
        super(ByteCNN, self).__init__()
        self.n = n
        self.emsize = emsize
        self.batch_norm = batch_norm
        self.encoder = ByteCNNEncoder(n, emsize, batch_norm=batch_norm,
                padding_idx=(ignore_index if ignore_index >= 0 else None),
                linear_layers=linear_layers,
                compress_channels=compress_channels)
        self.decoder = ByteCNNDecoder(n, emsize, batch_norm=batch_norm,
                                      linear_layers=linear_layers,
                                      compress_channels=compress_channels,
                                      output_embeddings_init=(self.encoder.embedding if output_embeddings else None))
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

    def _encode_decode(self, src, tgt, r_tgt=None):
        r_src = self.num_recurrences(src)
        r_tgt = self.num_recurrences(tgt) if r_tgt is None else r_tgt
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

    def try_on(self, batch_iterator, switch_to_evalmode=True, r_tgt=None):
        self.eval() if switch_to_evalmode else self.train()
        predicted = []
        for (src, tgt) in batch_iterator:
            src = Variable(src, volatile=True)
            tgt = Variable(tgt, volatile=True)
            decoded = self._encode_decode(src, tgt, r_tgt=r_tgt)
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


class NonRecurrentByteCNNEncoder(nn.Module):
    def __init__(self, n, emsize, input_len, batch_norm, batch_norm_eval_updates=False,
            padding_idx=None):
        super(NonRecurrentByteCNNEncoder, self).__init__()
        self.n = n
        self.emsize = emsize
        assert n % 2 == 0, 'n should be a multiple of 2'
        conv_kwargs = dict(kernel_size=3, stride=1, padding=1, bias=False)
        conv_proto = lambda: nn.Conv1d(emsize, emsize, **conv_kwargs)
        linear_proto = lambda: nn.Linear(emsize*4, emsize*4)
        residual_list = lambda proto, k: [Residual(proto, batch_norm=batch_norm) \
                                          for _ in xrange(k)]

        assert 2**int(np.log2(input_len)) == input_len
        n_nonrecurrent = int(np.log2(input_len) - 2)
        self.n_nonrecurrent = n_nonrecurrent

        self.embedding = nn.Embedding(emsize, emsize, padding_idx=padding_idx)
        self.prefix = nn.Sequential(*(residual_list(conv_proto, n//2)))
        self.nonrecurrent = nn.Sequential(*[nn.Sequential(*(residual_list(conv_proto, n//2) + [nn.MaxPool1d(kernel_size=2)])) \
                                            for _ in range(n_nonrecurrent)])
        self.postfix = nn.Sequential(*(residual_list(linear_proto, n//2-1) + \
                                       [Residual(linear_proto, out_relu=False, batch_norm=batch_norm)]))

        self.batch_norm = batch_norm
        self.batch_norm_eval_updates = batch_norm_eval_updates

    def forward(self, x):
        assert x.size(1) >= 4
        x = self.embedding(x).transpose(1, 2)
        x = self.prefix(x)
        x = self.nonrecurrent(x)
        bsz = x.size(0)
        return self.postfix(x.view(bsz, -1))


class NonRecurrentByteCNNDecoder(nn.Module):
    def __init__(self, n, emsize, input_len, batch_norm):
        super(NonRecurrentByteCNNDecoder, self).__init__()
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

        assert 2**int(np.log2(input_len)) == input_len
        n_nonrecurrent = int(np.log2(input_len) - 2)
        self.n_nonrecurrent = n_nonrecurrent

        self.prefix = nn.Sequential(*(residual_list(linear_proto, n//2)))
        if batch_norm:
            self.nonrecurrent = nn.Sequential(
                *[nn.Sequential(
                      *([expand_proto(), bn_proto(emsize), nn.ReLU(), conv_proto(),
                         bn_proto(emsize), nn.ReLU()] + \
                        residual_list(conv_proto, n//2-1)))
                  for _ in range(n_nonrecurrent)])
        else:
            self.nonrecurrent = nn.Sequential(
                *[nn.Sequential(
                      *([expand_proto(), nn.ReLU(), conv_proto(), nn.ReLU()] + \
                      residual_list(conv_proto, n_nonrecurrent//2-1)))
                  for _ in range(n_nonrecurrent)])
        self.postfix = nn.Sequential(*(residual_list(conv_proto, n//2)))

    def forward(self, x):
        x = self.prefix(x)
        x = x.view(x.size(0), self.emsize, 4)
        x = self.nonrecurrent(x)
        return self.postfix(x)


class NonRecurrentByteCNN(nn.Module):
    save_best = True
    def __init__(self, n=8, emsize=256, input_len=256, batch_norm=True,
                 ignore_index=-1, eos=0):  ## XXX Check default emsize
        super(NonRecurrentByteCNN, self).__init__()
        self.n = n
        self.emsize = emsize
        self.input_len = input_len
        self.batch_norm = batch_norm
        self.encoder = NonRecurrentByteCNNEncoder(n, emsize, input_len, batch_norm=batch_norm,
                padding_idx=(ignore_index if ignore_index >= 0 else None))
        self.decoder = NonRecurrentByteCNNDecoder(n, emsize, input_len, batch_norm=batch_norm)
        self.log_softmax = nn.LogSoftmax()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.eos = eos

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.log_softmax(x)

    def _encode_decode(self, src, tgt):
        features = self.encoder(src)
        return self.decoder(features)

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

        assert p['model_class'] == 'NonRecurrentByteCNN', \
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

    def get_state(self):
        return dict(kl_weight=self.kl_weight,
                    kl_increment_start=self.kl_increment_start)

    def load_state(self, state):
        self.kl_weight = state['kl_weight']
        if state.has_key('kl_increment_start'):
            self.kl_increment_start = state['kl_increment_start']

    def get_features_and_KL(self, mu, log_sigma):
        bs = mu.size(0)
        dim = mu.size(1)
        sigma = torch.exp(log_sigma)
        kl = -0.5 * torch.sum((1.0 + 2.0 * log_sigma - mu**2 - sigma**2) / (bs * dim))
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

    def _encode_decode(self, src, tgt, r_tgt=None, first_sample_random=False):
        r_src = self.num_recurrences(src)
        r_tgt = self.num_recurrences(tgt) if r_tgt is None else r_tgt
        dist_params = self.projection(self.encoder(src, r_src))
        mu, log_sigma = dist_params.chunk(2, dim=1)
        log_sigma /= 33.0
        features, kl = self.get_features_and_KL(mu, log_sigma)
        if first_sample_random:
            features[0] = torch.randn(*features[0].size())
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

    def try_on(self, batch_iterator, switch_to_evalmode=True, r_tgt=None,
               first_sample_random=False):
        self.eval() if switch_to_evalmode else self.train()
        predicted = []
        for batch, (src, tgt) in enumerate(batch_iterator):
            src = Variable(src, volatile=True)
            tgt = Variable(tgt, volatile=True)
            decoded, kl = self._encode_decode(src, tgt, r_tgt, first_sample_random=first_sample_random)
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
