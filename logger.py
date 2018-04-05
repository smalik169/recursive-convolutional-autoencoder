# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
# and https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
import datetime
import glob
import itertools
import os
import shutil
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf


def to_np(x):
    return x.data.cpu().numpy()

def print_model_summary(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model summary:\n%s" % (model,))
    print("Model params:\n%s" % ("\n".join(
        ["%s: %s" % (p[0], p[1].size()) for p in model.named_parameters()])))
    print("Number of params: %.2fM" % (num_params / 10.0**6))

def parse_resume_training(args):
    resume_path = args.resume_training
    print('\nLoading training state of %s' % resume_path)
    state = Logger.load_training_state(resume_path)
    state['args'].__dict__['resume_training'] = resume_path # XXX

    forced_args = None
    if args.resume_training_force_args != '':
        forced_args = eval('dict(%s)' % args.resume_training_force_args)
        print('\nForcing args: %s' % forced_args)
        print('\nWarning: Some args (e.g., --optimizer-kwargs) will be ignored. '
              'Some loaded components, as the optimizer, are already constructed.')
        for k,v in forced_args.items():
            if not hasattr(state['args'], k):
                print('\nWARNING: Setting arg which was previously unset: %s\n' % k)
            setattr(state['args'], k, v)
    state['forced_args'] = forced_args
    if args.resume_training_unroll:
        state['unroll_now'] = True

    # If some defaults are absent in state['args']
    for k,v in args.__dict__.items():
        if k not in state['args']:
            setattr(state['args'], k, v)

    state['forced_model_state'] = None
    if args.resume_training_force_model_state != '':
        state['forced_model_state'] = eval('dict(%s)' % args.resume_training_force_model_state)

    args = state['args']

    if 'model_kwargs' in args.__dict__:
        args.__dict__['model_kwargs'] = BackwardCompat.model_kwargs_str(
            args.__dict__['model_kwargs'])

    assert args.resume_training

    curr_logdir = args.resume_training.rstrip('/') + '/'
    if state['logger'].logdir != curr_logdir:
        print('WARNING: logdir changed: %s to %s' % (state['logger'].logdir, curr_logdir))
        state['logger'].logdir = curr_logdir

    print(args)
    print('\nWARNING: Ignoring other input arguments!\n')

    return args, state

def resume_training_innards(training_state, model, optimizer, scheduler):

    # State has been loaded before model construction
    logger = training_state['logger']
    # state = logger.set_training_state(training_state, optimizer, model)

    # Load model state
    model_state = training_state.get('model_state', None)
    forced_model_state = training_state.get('forced_model_state', None)
    if model_state:
        model.load_state(model_state)
    if forced_model_state:
        print('Forcing model state: %s' % forced_model_state)
        model.load_state(forced_model_state)

    state_dict = logger.load_model_state_dict(current=True)
    model.load_state_dict(state_dict, strict=True)

    # Load optimizer parameters
    optimizer.load_state_dict(training_state['optimizer'])
    # https://discuss.pytorch.org/t/saving-and-loading-sgd-optimizer/2536
    optimizer.state = defaultdict(dict, optimizer.state)

    if training_state.get('unroll_now', False):
        # Determine r
        data_kwargs = eval('dict(%s)' % training_state['args'].data_kwargs)
        r = int(np.log2(data_kwargs['fixed_len'])) - 2
        print('Unrolling the model')
        model.unroll(r, clone_weights=True)
        print_model_summary(model)

        # Add unrolled parameters to optimizer.
        # encoder.recurrent is of form nn.Sequential(old_sequential, new_sequential1, ...)
        # Only new_sequential parameters need to be added
        for name in ('encoder', 'decoder'):
            module = getattr(model, name).recurrent
            assert type(module) is nn.Sequential
            assert type(module[0]) is nn.Sequential
            sub_sequential_iter = iter(module)
            sub_sequential_iter.next()  # Skip old_sequential
            params = itertools.chain(*[l.parameters() for l in sub_sequential_iter])
            optimizer.add_param_group({'params': params})

    # Parse some of forced_args
    forced_args = training_state['forced_args']
    if forced_args and forced_args.has_key('lr'):
        optimizer.param_groups[0]['lr'] = forced_args['lr']
        logger.lr = forced_args['lr']

    first_epoch = logger.epoch + 1

    # Advance lr scheduler (it doesn't have load/save state_dict methods)
    if scheduler is not None:
        old_lr = optimizer.param_groups[0]['lr']
        for _ in range(1, first_epoch + 1):
            scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        logger.lr = new_lr
        print('Decaying lr_rate to epoch %d: %.5f --> %.5f' %
              (first_epoch, old_lr, new_lr))

    # Lastly set rng
    th = torch.cuda if training_state['args'].cuda else torch
    th.set_rng_state(training_state['random'])
    del training_state['random']

    return dict(model=model, optimizer=optimizer, scheduler=scheduler,
                logger=logger, first_epoch=first_epoch)


class BackwardCompat(object):

    @staticmethod
    def model_kwargs_str(kw_str):
        model_kwargs = eval('dict(%s)' % kw_str)
        # Old model kwargs: batch_norm=True, instance_norm=False,
        # New model kwargs: encoder_norm='batch' | 'instance' | None
        #                   decoder_norm='batch' | 'instance' | None
        was_bn = model_kwargs.get('batch_norm', False)
        was_in = model_kwargs.get('instance_norm', False)
        assert not (was_bn and was_in)

        if was_bn:
            extra_kwargs = dict(encode_norm='batch', decoder_norm='batch')
        elif was_in:
            extra_kwargs = dict(encode_norm='instance', decoder_norm='instance')
        else:
            extra_kwargs = {}

        if model_kwargs.has_key('batch_norm'):
            del model_kwargs['batch_norm']
        if model_kwargs.has_key('instance_norm'):
            del model_kwargs['instance_norm']
        model_kwargs.update(extra_kwargs)

        kw_str = ','.join('%s=%s' % kv for kv in model_kwargs.items())
        return kw_str


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
                        for name in ['train', 'sanity', 'valid', 'test']}
        self.total_losses = None
        self.num_samples = None
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
        self.total_losses = None

    def save_model_state_dict(self, model_state_dict, current=False):
        path = self.model_path if not current else self.current_model_path
        with open(path, 'wb') as f:
            torch.save(model_state_dict, f)

    def load_model_state_dict(self, path=None, current=False):
        if path is None:
            path = self.model_path if not current else self.current_model_path
        with open(path, 'rb') as f:
            return torch.load(f)

    def save_training_state(self, optimizer, args, model_state=None):
        th = torch.cuda if args.cuda else torch
        # XXX Writers cannot be pickled -- are they stateful or stateless?
        _writers = self.writers
        self.writers = None
        state = {'random': th.get_rng_state(),
                 'optimizer': optimizer.state_dict(),
                 'args': args,
                 'logger': self,
                 'model_state': model_state,
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

    def train_log(self, batch, batch_losses, named_params, num_samples):

        # logger.train_log(batch, {'nll_per_w': nll.data[0]},
        #                              named_params=self.named_parameters)
        # if log_every and batch % log_every == 0:
        #     print("Minibatch {0: >3}  | loss {1: >5.2f} | err rate {2: >5.2f}%" \
        #           .format(batch, losses[-1], err_rate))

        if self.total_losses is None:
            self.total_losses = dict(batch_losses)
            self.num_samples = num_samples
        else:
            for k, v in batch_losses.iteritems():
                self.total_losses[k] += v
            self.num_samples += num_samples

        if batch % self.log_interval == 0 and batch > 0:
            elapsed = (time.time() - self.minibatch_start_time
                       ) * 1000 / self.log_interval
            cur_loss = {k: v / self.num_samples #self.log_interval
                        for k, v in self.total_losses.items()}
            if 'err' in cur_loss:
                cur_loss['acc'] = 100. - cur_loss['err']
                del cur_loss['err']
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
            self.num_samples = None
            self.minibatch_start_time = time.time()

    def valid_log(self, val_loss, batch=0, mode='valid'):
        elapsed = time.time() - self.epoch_start_time
        losses = dict(val_loss)
        # losses['pplx'] = np.exp(val_loss['nll_per_w'])

        loss_str = mode + ' ' + ' | '.join(
            [' {} {:5.2f}'.format(k, v) for k, v in losses.items()])
        loss_str = ('| end of epoch {:3d} | time: {:5.2f}s | '
                    '{}'.format(self.epoch, elapsed, loss_str))
        print('-' * len(loss_str))
        print(loss_str)
        print('-' * len(loss_str))

        losses['s/epoch'] = elapsed
        losses['learning_rate'] = self.lr
        step = self.epoch * self.num_batches + batch
        self.tb_log(mode=mode, info=losses,
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
