

import sys
import os
import torch
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
from common import *
import tensorboard
from tensorboard import summary
from optparse import OptionParser
from clf_models import ResNet18
from torchvision.models.resnet import resnet18
import tensorboard_logger
parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--log', type=str, help="log path")
parser.add_option('--pretrained', type=int, help="log path", default=0)


def main(argv):
  (opts, args) = parser.parse_args(argv)
  if opts.gpu != -1:
    torch.cuda.set_device(opts.gpu)
  iterations = 0
 
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  if not os.path.exists(config.snapshot_prefix):
    os.makedirs(config.snapshot_prefix) 
  model_path = os.path.join(config.snapshot_prefix, 'model')
  log_path = os.path.join(config.snapshot_prefix, 'logs')
  if not os.path.exists(model_path):
    os.makedirs(model_path) 
  if not os.path.exists(log_path):
    os.makedirs(log_path) 
  tensorboard_logger.configure(log_path)



  batch_size = config.hyperparameters['batch_size']
  max_iterations = config.hyperparameters['max_iterations']
  lr = config.hyperparameters['lr']
  #model = resnet18(False, num_classes=2)
  #model = resnet18(opts.pretrained, num_classes=len(config.datasets['train_a']['list_name']))
  model = ResNet18(4)#vgg11_bn(opts.pretrained, num_classes=len(config.datasets['train_a']['list_name']))
  if opts.gpu != -1:
    model.cuda()
  print(model)
  crit = torch.nn.CrossEntropyLoss()
  opti = torch.optim.Adam(model.parameters(), lr=lr)
  train_loader = get_data_loader(config.datasets['train_a'], batch_size, num_workers=2)


  for ep in range(0, max_iterations):
      for im, label in train_loader:
        if im.size(0) != batch_size:
          continue
        im = Variable(im)
        label = Variable(label)
        if opts.gpu != -1:
            im, label = im.cuda(), label.cuda()
        model.zero_grad()
        pred = model(im)
        loss = crit(pred, label)
        loss.backward()
        opti.step()

        if (iterations + 1) % config.display == 0:
          print('Train Epoch: {} \t Iter: {} \tLoss: {:.6f}'.format(
                ep, iterations, loss.data[0]))
          tensorboard_logger.log_value('train_loss', loss.data[0], iterations)

        if (iterations+1) % config.snapshot_save_iterations == 0:
            torch.save(model.state_dict(), os.path.join(model_path, "%d.pth"%iterations))
        iterations += 1
        if iterations >= max_iterations:
            return
      
              
      lr = lr*(0.1**int(ep/10))      
      opti = torch.optim.Adam(model.parameters(), lr=lr)
      

if __name__ == '__main__':
  main(sys.argv)
