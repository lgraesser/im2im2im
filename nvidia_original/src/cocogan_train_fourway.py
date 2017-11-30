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
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--warm_start', type=int,
                  help="warm start training. If yes provide two pairs of generators and discriminators?", default=0)
parser.add_option('--gen_ab', type=str, help="generator ab for warm start")
parser.add_option('--gen_cd', type=str, help="generator cd for warm start")
parser.add_option('--disc_ab', type=str,
                  help="discriminator ab for warm start")
parser.add_option('--disc_cd', type=str,
                  help="discriminator cd for warm start")
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--log', type=str, help="log path")

MAX_EPOCHS = 100000


def main(argv):
    (opts, args) = parser.parse_args(argv)

    # Load experiment setting
    assert isinstance(opts, object)
    config = NetConfig(opts.config)

    batch_size = config.hyperparameters['batch_size']
    max_iterations = config.hyperparameters['max_iterations']

    train_loader_a = get_data_loader(config.datasets['train_a'], batch_size)
    train_loader_b = get_data_loader(config.datasets['train_b'], batch_size)
    train_loader_c = get_data_loader(config.datasets['train_c'], batch_size)
    train_loader_d = get_data_loader(config.datasets['train_d'], batch_size)

    trainer = []
    exec("trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer'])
    # Check if resume training
    iterations = 0
    if opts.resume == 1:
        iterations = trainer.resume(config.snapshot_prefix)
    trainer.cuda(opts.gpu)
    if opts.warm_start == 1:
        # Load models
        gen_ab = None
        gen_cd = None
        dis_ab = None
        dis_cb = None
        exec('gen_ab = %s(config.hyperparameters[\'gen\'])' %
             config.hyperparameters['gen']['name'])
        exec('gen_cd = %s(config.hyperparameters[\'gen\'])' %
             config.hyperparameters['gen']['name'])
        exec('dis_ab = %s(config.hyperparameters[\'dis\'])' %
             config.hyperparameters['dis']['name'])
        exec('dis_cd = %s(config.hyperparameters[\'dis\'])' %
             config.hyperparameters['dis']['name'])
        print("============ GENERATOR AB ==============")
        print(gen_ab)
        print("============ GENERATOR CD ==============")
        print(gen_cd)
        print("============ DISCRIMINATOR AB ==============")
        print(dis_ab)
        print("============ DISCRIMINATOR CD ==============")
        print(dis_cd)
        dirname = os.path.dirname(config.snapshot_prefix)
        model_path = os.path.join(dirname, opts.gen_ab)
        gen_ab.load_state_dict(torch.load(model_path))
        print("Pre trained generator ab loaded from: {}".format(model_path))
        model_path = os.path.join(dirname, opts.gen_cd)
        gen_cd.load_state_dict(torch.load(model_path))
        print("Pre trained generator cd loaded from: {}".format(model_path))
        gen_ab.cuda(opts.gpu)
        gen_cd.cuda(opts.gpu)
        model_path = os.path.join(dirname, opts.dis_ab)
        dis_ab.load_state_dict(torch.load(model_path))
        print("Pre trained discriminaor ab loaded from: {}".format(model_path))
        model_path = os.path.join(dirname, opts.dis_cd)
        dis_cd.load_state_dict(torch.load(model_path))
        print("Pre trained generator cd loaded from: {}".format(model_path))
        dis_ab.cuda(opts.gpu)
        dis_cd.cuda(opts.gpu)

        # Warm start init
        trainer.dis.model_A = dis_ab.model_A
        trainer.dis.model_B = dis_ab.model_B
        trainer.dis.model_C = dis_cd.model_A
        trainer.dis.model_D = dis_cd.model_B

        trainer.gen.encode_A = gen_ab.encode_A
        trainer.gen.encode_B = gen_ab.encode_B
        trainer.gen.encode_C = gen_cd.encode_A
        trainer.gen.encode_D = gen_cd.encode_B
        trainer.gen.decode_A = gen_ab.decode_A
        trainer.gen.decode_B = gen_ab.decode_B
        trainer.gen.decode_C = gen_cd.decode_A
        trainer.gen.decode_D = gen_cd.decode_B

        # Shared blocks - take mean of two original models
        # Functions inspired from this thread
        # https://discuss.pytorch.org/t/running-average-of-parameters/902/2
        def flatten_params(model1, model2):
            p1 =  torch.cat([param.data.view(-1) for param in model1.parameters()], 0)
            p2 =  torch.cat([param.data.view(-1) for param in model2.parameters()], 0)
            return (p1, p2)

        def load_params(flattened_params, model):
            offset = 0
            for param in model.parameters():
                param.data.copy_(
                    (flattened_params[0][offset:offset + param.nelement()] +
                    flattened_params[1][offset:offset + param.nelement()]) / 2.0
                    ).view(param.size())
                offset += param.nelement()

        model_S_new = flatten_params(dis_ab.model_S, dis_cd.model_S)
        load_params(model_S_new, trainer.dis.model_S)
        gen_enc_new = flatten_params(gen_ab.enc_shared, gen_cd.enc_shared)
        load_params(gen_enc_new, trainer.gen.enc_shared)
        gen_dec_new = flatten_params(gen_ab.dec_shared, gen_cd.dec_shared)
        load_params(gen_dec_new, trainer.gen.dec_shared)
        print("Initialized model with params from separately trained models")

    # print("============ DISCRIMINATOR ==============")
    # print(trainer.dis)
    # print("============ GENERATOR ==============")
    # print(trainer.gen)

    ######################################################################################################################
    # Setup logger and repare image outputs
    train_writer = tensorboard.FileWriter(
        "%s/%s" % (opts.log, os.path.splitext(os.path.basename(opts.config))[0]))
    image_directory, snapshot_directory = prepare_snapshot_and_image_folder(
        config.snapshot_prefix, iterations, config.image_save_iterations)

    for ep in range(0, MAX_EPOCHS):
        for it, (images_a, images_b, images_c, images_d) in enumerate(itertools.izip(train_loader_a, train_loader_b, train_loader_c, train_loader_d)):
            if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size or images_d.size(0) != batch_size:
                continue
            images_a = Variable(images_a.cuda(opts.gpu))
            images_b = Variable(images_b.cuda(opts.gpu))
            images_c = Variable(images_c.cuda(opts.gpu))
            images_d = Variable(images_d.cuda(opts.gpu))

            # Main training code
            trainer.dis_update(images_a, images_b, images_c,
                               images_d, config.hyperparameters)
            image_outputs = trainer.gen_update(
                images_a, images_b, images_c, images_d, config.hyperparameters)
            assembled_images = trainer.assemble_outputs(
                images_a, images_b, images_c, images_d, image_outputs)
            assembled_dbl_loop_images = trainer.assemble_double_loop_outputs(
                images_a, images_b, images_c, images_d, image_outputs)
            # print(assembled_images.data.shape)
            # print(assembled_dbl_loop_images.data.shape)

            # Dump training stats in log file
            if (iterations + 1) % config.display == 0:
                write_loss(iterations, max_iterations, trainer, train_writer)

            if (iterations + 1) % config.image_save_iterations == 0:
                img_filename = '%s/gen_%08d.jpg' % (
                    image_directory, iterations + 1)
                torchvision.utils.save_image(
                    assembled_images.data / 2 + 0.5, img_filename, nrow=2)
                dbl_img_filename = '%s/gen_dbl_%08d.jpg' % (
                    image_directory, iterations + 1)
                torchvision.utils.save_image(
                    assembled_dbl_loop_images.data / 2 + 0.5, dbl_img_filename, nrow=2)
                write_html(snapshot_directory + "/index.html", iterations +
                           1, config.image_save_iterations, image_directory)
            elif (iterations + 1) % config.image_display_iterations == 0:
                img_filename = '%s/gen.jpg' % (image_directory)
                torchvision.utils.save_image(
                    assembled_images.data / 2 + 0.5, img_filename, nrow=2)
                dbl_img_filename = '%s/gen_dbl.jpg' % (image_directory)
                torchvision.utils.save_image(
                    assembled_dbl_loop_images.data / 2 + 0.5, dbl_img_filename, nrow=2)

            # Save network weights
            if (iterations + 1) % config.snapshot_save_iterations == 0:
                trainer.save(config.snapshot_prefix, iterations)

            iterations += 1
            if iterations >= max_iterations:
                return


if __name__ == '__main__':
    main(sys.argv)
