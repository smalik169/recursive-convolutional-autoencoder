import os
import sys
from pprint import pprint
import tabulate

import torch

import logger


def model_details(model_dir):
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
    return details

def print_model_summary():
    dirs = [d for d in os.listdir('.') if d.startswith('2018-')]
    dirs = sorted(dirs, key=lambda r: r[1:])
    detail_dicts = [(d, model_details(d)) for d in dirs]
    rows = [[d['name'], d['data'], d['model'], d['file'], d['epoch'], d['args']] \
            for (name,d) in detail_dicts if d is not None]
    print tabulate.tabulate(rows)
    print '-' * 20
    print 'Broken experiments:'
    for name,d in detail_dicts:
        if d is None:
            print name

# print_model_summary()

for model_dir in sys.argv[1:]:
    details = model_details(model_dir)
    if details is None:
        print '<EMPTY>'
    else:
        pprint(details)
