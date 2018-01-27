# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
# and https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
import datetime
import glob
import os
import shutil
import sys
import time
from collections import defaultdict

import torch
import numpy as np
import tensorflow as tf


def to_np(x):
    return x.data.cpu().numpy()


class Writer(object):
    def __init__(self, logdir):
        """Create a summary writer logging to log_dir."""
        self.__writer = tf.summary.FileWriter(logdir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.__writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.__writer.add_summary(summary, step)

    def flush(self):
        self.__writer.flush()


class Logger(object):

    def __init__(self, initial_lr, log_interval, num_batches, logdir=None,
                 log_weights=False, log_grads=False):
        """Create a training logger with summary writers logging to logdir."""
        self.timestamp = datetime.datetime.now().isoformat().replace(':', '.')

        if not logdir:
            logdir = self.timestamp

        self.logdir = logdir.rstrip('/') + '/'
        os.makedirs(self.logdir)

        for py_file in glob.glob(r"*.py"):
            shutil.copy(py_file, self.logdir)

        self.model_path = self.logdir + "model.pt"
        self.current_model_path = self.logdir + "current_model.pt"
        self.writers = {name: Writer(self.logdir + name)
                        for name in ['train', 'valid', 'test']}
        self.total_losses = None
        self.epoch = 0
        self.lr = initial_lr
        self.log_interval = log_interval
        self.num_batches = num_batches
        self.log_weights = log_weights
        self.log_grads = log_grads
        self.minibatch_start_time = None
        self.epoch_start_time = None
        self.training_start_time = None

    def mark_epoch_start(self, epoch):
        self.epoch = epoch
        self.minibatch_start_time = self.epoch_start_time = time.time()
        self.total_losses = 0

    def save_model_state_dict(self, model_state_dict, current=False):
        path = self.model_path if not current else self.current_model_path
        with open(path, 'wb') as f:
            torch.save(model_state_dict, f)

    def load_model_state_dict(self, current=False):
        path = self.model_path if not current else self.current_model_path
        with open(path, 'rb') as f:
            return torch.load(f)

    def save_training_state(self, optimizer, args):
        th = torch.cuda if args.cuda else torch
        # XXX Writers cannot be pickled -- are they stateful or stateless?
        _writers = self.writers
        self.writers = None
        state = {'random': th.get_rng_state(),
                 'optimizer': optimizer.state_dict(),
                 'args': args,
                 'logger': self,
                 }
        torch.save(state, self.training_state_path(self.logdir))
        self.writers = _writers

    @staticmethod
    def training_state_path(logdir):
        return os.path.join(logdir, 'training_state.pkl')

    @staticmethod
    def load_training_state(resume_path):
        state_path = Logger.training_state_path(resume_path)
        state = torch.load(open(state_path, 'rb'))
        # XXX Writers cannot be pickled -- are they stateful or stateless?
        state['logger'].writers = {
            name: Writer(state['logger'].logdir + name) \
            for name in ['train', 'valid', 'test']}
        return state

    def set_training_state(self, state, optimizer):
        th = torch.cuda if state['args'].cuda else torch
        th.set_rng_state(state['random'])
        del state['random']
        optimizer.load_state_dict(state['optimizer'])
        # https://discuss.pytorch.org/t/saving-and-loading-sgd-optimizer/2536
        optimizer.state = defaultdict(dict, optimizer.state)
        state['optimizer'] = optimizer
        return state

    def save_model_info(self, classes_with_kwargs):

        kwargs_to_str = lambda kwargs: ','.join(
            ["%s=%s" % (key, str(kw) if type(kw) != str else '\\"%s\\"' % kw) \
             for key,kw in kwargs.items()])

        info = ""
        for field, (name, kwargs) in classes_with_kwargs.items():
            info += "%s_class=%s\n" % (field, name)
            if kwargs:
                info += "%s_kwargs=%s\n" % (field, kwargs_to_str(kwargs)) 

        with open(self.logdir+"model.info", 'w') as f:
            f.write(info.strip())

    def train_log(self, batch, batch_losses, named_params):

        # logger.train_log(batch, {'nll_per_w': nll.data[0]},
        #                              named_params=self.named_parameters)
        # if log_every and batch % log_every == 0:
        #     print("Minibatch {0: >3}  | loss {1: >5.2f} | err rate {2: >5.2f}%" \
        #           .format(batch, losses[-1], err_rate))

        if not self.total_losses:
            self.total_losses = dict(batch_losses)
        else:
            for k, v in batch_losses.iteritems():
                self.total_losses[k] += v

        if batch % self.log_interval == 0 and batch > 0:
            elapsed = (time.time() - self.minibatch_start_time
                       ) * 1000 / self.log_interval
            cur_loss = {k: v / self.log_interval
                        for k, v in self.total_losses.items()}
            # cur_loss['pplx'] = np.exp(cur_loss['nll_per_w'])
            loss_str = ' | '.join(
                [' {} {:5.2f}'.format(k, cur_loss[k]) \
                 for k in sorted(cur_loss.keys())])
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | '
                  'ms/batch {:5.2f} | {}'.format(
                    self.epoch, batch, self.num_batches, self.lr,
                    elapsed, loss_str))

            cur_loss["ms/batch"] = elapsed
            cur_loss["learning_rate"] = self.lr

            step = (self.epoch-1) * self.num_batches + batch

            self.tb_log(mode="train", info=cur_loss, step=step,
                        named_params=named_params)

            self.total_losses = None
            self.minibatch_start_time = time.time()

    def valid_log(self, val_loss, batch=0):
        elapsed = time.time() - self.epoch_start_time
        losses = dict(val_loss)
        # losses['pplx'] = np.exp(val_loss['nll_per_w'])

        loss_str = ' : '.join(
            [' {} {:5.2f}'.format(k, v) for k, v in losses.items()])
        loss_str = ('| end of epoch {:3d} | time: {:5.2f}s | '
                    'valid {}'.format(self.epoch, elapsed, loss_str))
        print('-' * len(loss_str))
        print(loss_str)
        print('-' * len(loss_str))

        losses['s/epoch'] = elapsed
        losses['learning_rate'] = self.lr
        step = self.epoch * self.num_batches + batch
        self.tb_log(mode="valid", info=losses,
                    step=step, named_params=lambda: [])

    def mem_log(self, mode, named_params, batch):
        step = self.epoch * self.num_batches + batch
        for tag, value in named_params:
            self.writers[mode].histo_summary(tag, to_np(value), step, bins=20)
        # self.writers[mode].flush()

    def final_log(self, results, result_file="results/log_file.md"):
        if not os.path.exists(os.path.dirname(result_file)):
            os.makedirs(os.path.dirname(result_file))
        #for losses in results.values():
        #    losses['pplx'] = np.exp(losses['nll_per_w'])

        log_line = ('| End of training | test losses {} |'
                    ''.format(results['test']))
        print('=' * len(log_line))
        print(log_line)
        print('=' * len(log_line))

        header =  "|timestamp|args|train acc|valid acc|test acc|other|\n"
        header += "|---------|----|---------|---------|--------|-----|\n"

	if not results.has_key('train') or not results.has_key('valid'):
            log_line = "| %s | %s | not_evald | not_evald | %.2f | %s |\n" % (
                self.timestamp, '<br>'.join(sys.argv[1:]),
                results['test']['acc'], results)
	else:
            log_line = "| %s | %s | %.2f | %.2f | %.2f | %s |\n" % (
                self.timestamp, '<br>'.join(sys.argv[1:]),
                results['train']['acc'], results['valid']['pplx'],
                results['test']['acc'], results)

        with open(self.logdir+"results.md", 'w') as f:
            f.write(header + log_line)

        if not os.path.isfile(result_file):
            with open(result_file, 'a') as f:
                f.write(header)

        with open(result_file, 'a') as f:
            f.write(log_line)

        step = self.epoch * self.num_batches
        for mode in results.keys():
            self.tb_log(mode=mode, info=results[mode], step=step,
                        named_params=lambda: [])

    def tb_log(self, mode, info, step, named_params):
            # Log scalar values
            for tag, value in info.items():
                self.writers[mode].scalar_summary(tag, value, step)

            # Log values and gradients of the parameters (histogram)
            if self.log_weights:
                for tag, value in named_params():
                    tag = tag.replace('.', '/')
                    self.writers[mode].histo_summary(tag, to_np(value), step)

            if self.log_grads:
                for tag, value in named_params():
                    tag = tag.replace('.', '/')
                    self.writers[mode].histo_summary(tag+'/grad',
                                                     to_np(value.grad), step)

            self.writers[mode].flush()
