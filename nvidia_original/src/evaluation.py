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
from collections import Counter

parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--gen_ab', type=str, help="generator ab")
parser.add_option('--gen_cd', type=str, help="generator cd")
parser.add_option('--mode', type=str, help="generator cd", default="separate")
parser.add_option('--chunk', type=int, help="generator cd", default=0)
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
    clf.load_state_dict(torch.load(opts.clf_model_path))
    clf.cuda(opts.gpu)
    clf.eval()


    train_loader_a = get_data_loader(config.datasets['train_a'], batch_size, shuffle=False, num_workers=1)
    train_loader_b = get_data_loader(config.datasets['train_b'], batch_size, shuffle=False, num_workers=1)
    #train_loader_c = get_data_loader(config.datasets['train_c'], batch_size, shuffle=False, num_workers=1)
    #train_loader_d = get_data_loader(config.datasets['train_d'], batch_size, shuffle=False, num_workers=1)

    dirname = os.path.dirname(config.snapshot_prefix)

    if opts.mode == "separate":
        gen_ab = COCOResGen2(config.hyperparameters['gen_ab'])
        gen_cd = COCOResGen2(config.hyperparameters['gen_cd'])

        model_path = os.path.join(dirname, opts.gen_ab)
        gen_ab.load_state_dict(torch.load(model_path))
        model_path = os.path.join(dirname, opts.gen_cd)
        gen_cd.load_state_dict(torch.load(model_path))
        gen_ab.cuda(opts.gpu)
        gen_cd.cuda(opts.gpu)
    elif opts.mode == "joint":
        gen_ab = COCOResGenSmallK4Way(config.hyperparameters['gen_ab']) 
        model_path = os.path.join(dirname, opts.gen_ab)
        saved_model = torch.load(model_path, map_location=lambda storage, loc: storage)
        gen_ab.load_state_dict(saved_model)
        gen_ab.cuda(opts.gpu)

    '''
    exec('gen_ab = %s(config.hyperparameters[\'gen_ab\'])' %
         config.hyperparameters['gen_ab']['name'])
    exec('gen_cd = %s(config.hyperparameters[\'gen_cd\'])' %
         config.hyperparameters['gen_cd']['name'])
    '''

    print("Pre trained generator ab loaded")
    print("Pre trained generator cd loaded")
    it = 0
    image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, it, config.image_save_iterations)
    true_labels = torch.LongTensor(batch_size).cuda(opts.gpu)

    steps = {0: Counter(), 1: Counter(), 2: Counter(), 3:Counter()}

    def classify_samples(x, y, step):
        outputs = clf(x)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(y).cpu().sum()
        steps[step][predicted.cpu().numpy()[0]] +=1
        return correct


    #for it, (images_a, images_b, images_c, images_d) in enumerate(itertools.izip(train_loader_a, train_loader_b, train_loader_c, train_loader_d)):
    #    if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size or images_d.size(0) != batch_size:
    #        continue
    correct_blond, correct_brunette, correct_brunette_smiling, correct_blond_smiling = 0, 0, 0, 0
    translations = []
    to_smiling = []
    for it, (images_a, images_b) in enumerate(itertools.izip(train_loader_a, train_loader_b)):
        if it >= max_iterations:
            break
        if images_a.size(0) != batch_size or images_b.size(0) != batch_size:
            continue

        images_a = Variable(images_a.cuda(opts.gpu))
        images_b = Variable(images_b.cuda(opts.gpu))
        #images_c = Variable(images_c.cuda(opts.gpu))
        #images_d = Variable(images_d.cuda(opts.gpu))

        if classify_samples(images_a, true_labels.fill_(0), 0) == 1:
            correct_blond += 1
        else:
            continue

        gen_ab.eval()
        if opts.mode == "joint":
            x_ab, _ = gen_ab.forward_a2b(images_a)
        else:
            x_aa, x_ba, x_ab, x_bb, _ = gen_ab(images_a, images_b)        
        x_ab = normalize_image(x_ab)
        if classify_samples(x_ab, true_labels.fill_(1), 1) == 1:
            correct_brunette += 1
        else:
            continue

        if opts.mode == "joint":
            x_ab_dc, _ = gen_ab.forward_d2c(x_ab)
        else:
            gen_cd.eval()        
            x_ab_dc, _ = gen_cd.forward_b2a(x_ab)
                
        to_smiling.append(torch.cat((images_a, x_ab, x_ab_dc), 3))
        x_ab_dc = normalize_image(x_ab_dc)
        if classify_samples(x_ab_dc, true_labels.fill_(3), 2) == 1:
            correct_brunette_smiling += 1
        else:
            continue

        if opts.mode == "joint":
            x_ab_dc_a, _ = gen_ab.forward_b2a(x_ab_dc)
        else:
            x_ab_dc_a, _ = gen_ab.forward_b2a(x_ab_dc)
        x_ab_dc_a = normalize_image(x_ab_dc_a)
        if classify_samples(x_ab_dc_a, true_labels.fill_(2), 3) == 1:
            correct_blond_smiling += 1
        else:
            continue

        translations.append(torch.cat((images_a, x_ab, x_ab_dc, x_ab_dc_a), 3))
    print("Total Images: %d \nBlonde: %d \nBrunette: %d \nSmiling_Brunette: %d \nSmiling_Blonde: %d"%(max_iterations, correct_blond, correct_brunette, correct_brunette_smiling, correct_blond_smiling)) 
    if len(translations) == 0:
        translations = to_smiling
    assembled_images = torch.cat(translations, 0)
    print(assembled_images.size())

    if opts.chunk:
        for _, sample in enumerate(assembled_images.chunk(10, 0)):
            torchvision.utils.save_image(
                sample.data / 2 + 0.5, 'sample_%d.png'%_, nrow=1)
    else: 
        torchvision.utils.save_image(
            assembled_images.data / 2 + 0.5, 'sample2.png', nrow=1)
    print(steps)
if __name__ == '__main__':
    main(sys.argv)
