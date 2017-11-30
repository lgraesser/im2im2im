#!/usr/bin/env python

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import sys
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
from common import *
import tensorboard
from tensorboard import summary
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--gen_ab', type=str, help="generator ab")
parser.add_option('--gen_cd', type=str, help="generator cd")

def main(argv):
    (opts, args) = parser.parse_args(argv)

    # Load experiment setting
    assert isinstance(opts, object)
    config = NetConfig(opts.config)

    batch_size = config.hyperparameters['batch_size']

    train_loader_a = get_data_loader(config.datasets['train_a'], batch_size)
    train_loader_b = get_data_loader(config.datasets['train_b'], batch_size)
    train_loader_c = get_data_loader(config.datasets['train_c'], batch_size)
    train_loader_d = get_data_loader(config.datasets['train_d'], batch_size)

    gen_ab = None
    gen_bc = None
    exec('gen_ab = %s(config.hyperparameters[\'gen\'])' %
         config.hyperparameters['gen_ab']['name'])
    exec('gen_bc = %s(config.hyperparameters[\'gen\'])' %
         config.hyperparameters['gen_bc']['name'])
    dirname = os.path.dirname(snapshot_prefix)
    model_path = os.path.join(dirname, opts.gen_ab)
    gen_ab.load_state_dict(torch.load(model_path))
    model_path = os.path.join(dirname, opts.gen_cd)
    gen_cd.load_state_dict(torch.load(model_path))
    print("============ GENERATOR AB ==============")
    print(gen_ab)
    print("============ GENERATOR CD ==============")
    print(gen_bc)


if __name__ == '__main__':
    main(sys.argv)
