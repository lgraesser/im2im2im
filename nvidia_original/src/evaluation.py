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

from vgg_model import *

parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--gen_ab', type=str, help="generator ab")
parser.add_option('--gen_cd', type=str, help="generator cd")
parser.add_option('--clf_model_path', type=str, help="clf model path", default="")

def normalize_image(x):
  return x[:, 0:3, :, :]

def main(argv):
    (opts, args) = parser.parse_args(argv)

    # Load experiment setting
    assert isinstance(opts, object)
    config = NetConfig(opts.config)

    batch_size = config.hyperparameters['batch_size']
    max_iterations = config.hyperparameters['max_iterations']

    clf = VGG('VGG11', 4)
    clf.cuda(opts.gpu)
    clf.load_state_dict(torch.load(opts.clf_model_path))
    clf.eval()


    train_loader_a = get_data_loader(config.datasets['train_a'], batch_size, shuffle=False, num_workers=1)
    train_loader_b = get_data_loader(config.datasets['train_b'], batch_size, shuffle=False, num_workers=1)
    #train_loader_c = get_data_loader(config.datasets['train_c'], batch_size, shuffle=False, num_workers=1)
    #train_loader_d = get_data_loader(config.datasets['train_d'], batch_size, shuffle=False, num_workers=1)

    gen_ab = None
    gen_cd = None
    gen_ab = COCOResGen2(config.hyperparameters['gen_ab'])
    gen_cd = COCOResGen2(config.hyperparameters['gen_cd'])

    '''
    exec('gen_ab = %s(config.hyperparameters[\'gen_ab\'])' %
         config.hyperparameters['gen_ab']['name'])
    exec('gen_cd = %s(config.hyperparameters[\'gen_cd\'])' %
         config.hyperparameters['gen_cd']['name'])
    '''

    #print("============ GENERATOR AB ==============")
    #print(gen_ab)
    #print("============ GENERATOR CD ==============")
    #print(gen_cd)
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
    true_labels = torch.LongTensor(batch_size).cuda(opts.gpu)


    def classify_samples(x, y):
        outputs = clf(x)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(y).cpu().sum()
        return correct


    #for it, (images_a, images_b, images_c, images_d) in enumerate(itertools.izip(train_loader_a, train_loader_b, train_loader_c, train_loader_d)):
    #    if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size or images_d.size(0) != batch_size:
    #        continue
    correct_blond, correct_brunette, correct_brunette_smiling, correct_blond_smiling = 0, 0, 0, 0
    translations = []
    for it, (images_a, images_b) in enumerate(itertools.izip(train_loader_a, train_loader_b)):
        if it >= max_iterations:
            break
        if images_a.size(0) != batch_size or images_b.size(0) != batch_size:
            continue

        images_a = Variable(images_a.cuda(opts.gpu))
        images_b = Variable(images_b.cuda(opts.gpu))
        #images_c = Variable(images_c.cuda(opts.gpu))
        #images_d = Variable(images_d.cuda(opts.gpu))

        if classify_samples(images_a, true_labels.fill_(0)) == 1:
            correct_blond += 1
        else:
            continue

        gen_ab.eval()
        x_aa, x_ba, x_ab, x_bb, _ = gen_ab(images_a, images_b)        
        x_ab = normalize_image(x_ab)
        if classify_samples(x_ab, true_labels.fill_(1)) == 1:
            correct_brunette += 1
        else:
            continue

        gen_cd.eval()        
        x_ab_dc, _ = gen_cd.forward_b2a(x_ab)
        x_ab_dc = normalize_image(x_ab_dc)
        if classify_samples(x_ab_dc, true_labels.fill_(3)) == 1:
            correct_brunette_smiling += 1
        else:
            continue

        x_ab_dc_a, _ = gen_ab.forward_b2a(x_ab_dc)
        x_ab_dc_a = normalize_image(x_ab_dc_a)
        if classify_samples(x_ab_dc_a, true_labels.fill_(2)) == 1:
            correct_blond_smiling += 1
        else:
            continue

        translations.append(torch.cat((images_a, x_ab, x_ab_dc, x_ab_dc_a), 3))
    print("Total Images: %d \nBlonde: %d \nBrunette: %d \nSmiling_Brunette: %d \nSmiling_Blonde: %d"%(max_iterations, correct_blond, correct_brunette, correct_brunette_smiling, correct_blond_smiling)) 
    assembled_images = torch.cat(translations, 0)
    print(assembled_images.size())

    torchvision.utils.save_image(
        assembled_images.data / 2 + 0.5, 'sample.png', nrow=1)
        # assembled_dbl_loop_images =  torch.cat((
        #         images_a[0:1, ::], x_ab[0:1, ::], x_ab_cd[0:1, ::], x_ab_dc[0:1, ::],
        #         images_b[0:1, ::], x_ba[0:1, ::], x_ba_cd[0:1, ::], x_ba_dc[0:1, ::],
        #         images_c[0:1, ::], x_cd[0:1, ::], x_cd_ab[0:1, ::], x_cd_ba[0:1, ::],
        #         images_d[0:1, ::], x_dc[0:1, ::], x_dc_ab[0:1, ::], x_dc_ba[0:1, ::]
        #         ), 3)

        # # save images
        # if (it + 1) % config.image_save_iterations == 0:
        #     img_filename = '%s/gen_%08d.jpg' % (
        #         image_directory, it + 1)
        #     dbl_img_filename = '%s/gen_dbl_%08d.jpg' % (
        #         image_directory, it + 1)
        #     torchvision.utils.save_image(
        #         assembled_dbl_loop_images.data / 2 + 0.5, dbl_img_filename, nrow=2)
        # if (it + 1) % 10 == 0:
        #   print("Iteration: {}".format(it + 1))

if __name__ == '__main__':
    main(sys.argv)
