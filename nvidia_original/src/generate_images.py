#!/usr/bin/env python

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

'''
Takes a four way trainer and generates composed image to image translation
across four distributions
'''
import sys
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
from common import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--gen', type=str, help="pre-trained 4 way generator")
parser.add_option('--dis', type=str, help="pre-trained 4 way discriminator")

def normalize_image(x):
  return x[:, 0:3, :, :]

def main(argv):
    (opts, args) = parser.parse_args(argv)

    # Load experiment setting
    assert isinstance(opts, object)
    config = NetConfig(opts.config)

    batch_size = config.hyperparameters['batch_size']

    train_loader_a = get_data_loader(config.datasets['train_a'], batch_size, shuffle=False, num_workers=1)
    train_loader_b = get_data_loader(config.datasets['train_b'], batch_size, shuffle=False, num_workers=1)
    train_loader_c = get_data_loader(config.datasets['train_c'], batch_size, shuffle=False, num_workers=1)
    train_loader_d = get_data_loader(config.datasets['train_d'], batch_size, shuffle=False, num_workers=1)

    trainer = []
    exec ("trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer'])
    print("============ GENERATOR AB ==============")
    print(trainer.gen)
    print("============ GENERATOR CD ==============")
    print(trainer.dis)
    dirname = os.path.dirname(config.snapshot_prefix)
    model_path = os.path.join(dirname, opts.gen)
    trainer.gen.load_state_dict(torch.load(model_path))
    print("Pre trained generator loaded")
    model_path = os.path.join(dirname, opts.dis)
    trainer.dis.load_state_dict(torch.load(model_path))
    print("Pre trained discriminator loaded")
    trainer.gen.cuda(opts.gpu)
    trainer.dis.cuda(opts.gpu)

    it = 0
    image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, it, config.image_save_iterations)

    for it, (images_a, images_b, images_c, images_d) in enumerate(itertools.izip(train_loader_a, train_loader_b, train_loader_c, train_loader_d)):
        if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size or images_d.size(0) != batch_size:
            continue
        images_a = Variable(images_a.cuda(opts.gpu))
        images_b = Variable(images_b.cuda(opts.gpu))
        images_c = Variable(images_c.cuda(opts.gpu))
        images_d = Variable(images_d.cuda(opts.gpu))

        # Get image outputs
        trainer.gen.eval()
        trainer.dis.eval()
        x_aa, x_ba, x_ab, x_bb, x_cc, x_dc, x_cd, x_dd, shared = \
            trainer.gen(images_a, images_b, images_c, images_d)
        x_bab, shared_bab = trainer.gen.forward_a2b(x_ba)
        x_aba, shared_aba = trainer.gen.forward_b2a(x_ab)
        x_dcd, shared_dcd = trainer.gen.forward_c2d(x_dc)
        x_cdc, shared_cdc = trainer.gen.forward_d2c(x_cd)
        image_outputs = (x_aa, x_ba, x_ab, x_bb, x_aba, x_bab, x_cc, x_dc, x_cd, x_dd, x_cdc, x_dcd)

        assembled_images = trainer.assemble_outputs(
            images_a, images_b, images_c, images_d, image_outputs)
        assembled_dbl_loop_images = trainer.assemble_double_loop_outputs(
            images_a, images_b, images_c, images_d, image_outputs)

        # save images
        if (it + 1) % config.image_save_iterations == 0:
            img_filename = '%s/gen_%08d.jpg' % (
                image_directory, it + 1)
            torchvision.utils.save_image(
                assembled_images.data / 2 + 0.5, img_filename, nrow=2)
            dbl_img_filename = '%s/gen_dbl_%08d.jpg' % (
                image_directory, it + 1)
            torchvision.utils.save_image(
                assembled_dbl_loop_images.data / 2 + 0.5, dbl_img_filename, nrow=2)
        if (it + 1) % 10 == 0:
          print("Iteration: {}".format(it + 1))

if __name__ == '__main__':
    main(sys.argv)
