#!/usr/bin/env python

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

'''
Takes a two, two way trainers and generates composed image to image translation
across four distributions
'''
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

    gen_ab = None
    gen_cd = None
    exec('gen_ab = %s(config.hyperparameters[\'gen_ab\'])' %
         config.hyperparameters['gen_ab']['name'])
    exec('gen_cd = %s(config.hyperparameters[\'gen_cd\'])' %
         config.hyperparameters['gen_cd']['name'])
    print("============ GENERATOR AB ==============")
    print(gen_ab)
    print("============ GENERATOR CD ==============")
    print(gen_cd)
    dirname = os.path.dirname(config.snapshot_prefix)
    model_path = os.path.join(dirname, opts.gen_ab)
    gen_ab.load_state_dict(torch.load(model_path))
    print("Pre trained generator ab loaded")
    model_path = os.path.join(dirname, opts.gen_cd)
    gen_cd.load_state_dict(torch.load(model_path))
    print("Pre trained generator cd loaded")
    gen_ab.cuda(opts.gpu)
    gen_cd.cuda(opts.gpu)
    it = 0
    image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, it, config.image_save_iterations)

    for it, (images_a, images_b, images_c, images_d) in enumerate(itertools.izip(train_loader_a, train_loader_b, train_loader_c, train_loader_d)):
        if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size or images_d.size(0) != batch_size:
            continue
        images_a = Variable(images_a.cuda(opts.gpu))
        images_b = Variable(images_b.cuda(opts.gpu))
        images_c = Variable(images_c.cuda(opts.gpu))
        images_d = Variable(images_d.cuda(opts.gpu))

        # Single loop pass
        gen_ab.eval()
        x_aa, x_ba, x_ab, x_bb, _ = gen_ab(images_a, images_b)
        x_bab, _ = gen_ab.forward_a2b(x_ba)
        x_aba, _ = gen_ab.forward_b2a(x_ab)

        gen_cd.eval()
        x_cc, x_dc, x_cd, x_dd, _ = gen_cd(images_c, images_d)
        x_dcd, _ = gen_cd.forward_a2b(x_dc)
        x_cdc, _ = gen_cd.forward_b2a(x_cd)

        # Double loop pass
        x_cd_ab, _ = gen_ab.forward_a2b(x_cd)
        x_cd_ba, _ = gen_ab.forward_b2a(x_cd)
        x_dc_ab, _ = gen_ab.forward_a2b(x_dc)
        x_dc_ba, _ = gen_ab.forward_b2a(x_dc)

        x_ab_cd, _ = gen_cd.forward_a2b(x_ab)
        x_ab_dc, _ = gen_cd.forward_b2a(x_ab)
        x_ba_cd, _ = gen_cd.forward_a2b(x_ba)
        x_ba_dc, _ = gen_cd.forward_b2a(x_ba)

        # Assemble images
        # Single loop
        images_a = normalize_image(images_a)
        images_b = normalize_image(images_b)
        images_c = normalize_image(images_c)
        images_d = normalize_image(images_d)
        x_aa = normalize_image(x_aa)
        x_ba = normalize_image(x_ba)
        x_ab = normalize_image(x_ab)
        x_bb = normalize_image(x_bb)
        x_aba = normalize_image(x_aba)
        x_bab = normalize_image(x_bab)
        x_cc = normalize_image(x_cc)
        x_dc = normalize_image(x_dc)
        x_cd = normalize_image(x_cd)
        x_dd = normalize_image(x_dd)
        x_cdc = normalize_image(x_cdc)
        x_dcd = normalize_image(x_dcd)
        assembled_images =  torch.cat((
                images_a[0:1, ::], x_aa[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::],
                images_b[0:1, ::], x_bb[0:1, ::], x_ba[0:1, ::], x_bab[0:1, ::],
                images_c[0:1, ::], x_cc[0:1, ::], x_cd[0:1, ::], x_cdc[0:1, ::],
                images_d[0:1, ::], x_dd[0:1, ::], x_dc[0:1, ::], x_dcd[0:1, ::]), 3)

        # double loop
        x_ab_cd = normalize_image(x_ab_cd)
        x_ab_dc = normalize_image(x_ab_dc)
        x_ba_cd = normalize_image(x_ba_cd)
        x_ba_dc = normalize_image(x_ba_dc)
        x_cd_ab = normalize_image(x_cd_ab)
        x_cd_ba = normalize_image(x_cd_ba)
        x_dc_ab = normalize_image(x_dc_ab)
        x_dc_ba = normalize_image(x_dc_ba)
        assembled_dbl_loop_images =  torch.cat((
                images_a[0:1, ::], x_ab[0:1, ::], x_ab_cd[0:1, ::], x_ab_dc[0:1, ::],
                images_b[0:1, ::], x_ba[0:1, ::], x_ba_cd[0:1, ::], x_ba_dc[0:1, ::],
                images_c[0:1, ::], x_cd[0:1, ::], x_cd_ab[0:1, ::], x_cd_ba[0:1, ::],
                images_d[0:1, ::], x_dc[0:1, ::], x_dc_ab[0:1, ::], x_dc_ba[0:1, ::]
                ), 3)

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
