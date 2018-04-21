import argparse
import os
import sys
from pprint import pprint

import tabulate
import numpy as np

import torch
import tensorflow as tf

import logger


def model_details(model_dir, accuracy=False):
    state_file = os.path.join(model_dir, 'training_state.pkl')
    if not os.path.isfile(state_file):
        return None
    state = logger.Logger.load_training_state(model_dir)
    details = dict(name=model_dir)
    log = state.get('logger', None)
    args = state['args'].__dict__
    details['epoch'] = log.epoch if log else '0'
    details['model'] = args.get('model', 'ByteCNN')
    details['data'] = args.get('data', '').split('/')[-2]
    details['file'] = args.get('file_class', 'UTF8File')
    details['args'] = args
    # print model_dir.split('/')[1], epoch, top_dir
   
    if accuracy:
        steps = load_exp_steps(model_dir)
        for s in ('train', 'valid'):
            try:
                val = steps[s][sorted(steps[s].keys())[-1]]['acc'][0]
                val = np.round(val, 2)
            except:
                val = 'UNK'
            details['acc_%s' % s] = val
    return details

def print_summary(exp_all, include_args=False, include_accuracy=False):
    exp_valid = [(name, d) for (name, d) in exp_all if d is not None]
    exp_empty = [(name, d) for (name, d) in exp_all if d is None]
    columns = ('name', 'data', 'model', 'file', 'epoch')
    if include_accuracy:
        columns += ('acc_valid', 'acc_train')
    if include_args:
        columns += ('args',)
    rows = [[d[col] for col in columns] for (name,d) in exp_valid]
    print tabulate.tabulate(rows)
    if exp_empty:
        print 'Empty experiments:'
        for (name, _) in exp_empty:
            print name

def parse_tf_steps(path):
    steps = {}
    for summary in tf.train.summary_iterator(path):
        values = summary.summary.value
        step = summary.step

        assert len(values) <= 1, summary
        if len(values) != 0:
            if not steps.has_key(step):
                steps[step] = {}
            summ = steps[step]

            tag = values[0].tag
            sv = values[0].simple_value
            if not summ.has_key(tag):
                summ[tag] = []
            summ[tag].append(sv)
    return steps

def load_split_steps(path, split):
    ret = {}
    path = os.path.join(path, split)
    for f in [os.path.join(path, f) for f in os.listdir(path)]:
        ret.update(parse_tf_steps(f))
    return ret

def load_exp_steps(path):
    return {s: load_split_steps(path, s) \
            for s in ('train', 'valid', 'test')}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Byte-level CNN model manager')
    parser.add_argument('dirs', type=str, nargs='+')
    parser.add_argument('--table', action='store_true', default=False,
                        help='Print summary table')
    parser.add_argument('--args', action='store_true', default=False,
                        help='Print all args for an experiment')
    parser.add_argument('--accuracy', action='store_true', default=False,
                        help='Report accuracy')
    args = parser.parse_args()

    # dirs = [d for d in os.listdir('.') if d.startswith('2018-')]
    exp_details = [(name, model_details(name, accuracy=args.accuracy)) for name in args.dirs]
    if args.table:
        print_summary(exp_details, include_args=args.args, include_accuracy=args.accuracy)
    else:
        for (name, d) in exp_details:
            print name
            if d is None:
                print '  <EMPTY>'
                continue
            del d['name']
            if not args.args:
                del d['args']
            pprint(d, indent=2, width=60)
