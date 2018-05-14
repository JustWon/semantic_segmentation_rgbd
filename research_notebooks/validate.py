import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

from SUNRGBDLoader import *
from NYUDv2Loader import *

torch.backends.cudnn.benchmark = True

cudnn.benchmark = True

def validate(args):

    model_name = args.model_name

    if (args.dataset == 'NYUDv2'):
        data_path = '/home/dongwonshin/Desktop/Datasets/NYUDv2/'
        t_loader = NYUDv2Loader(data_path, is_transform=True)
        v_loader = NYUDv2Loader(data_path, is_transform=True, split='val')
    elif (args.dataset == 'SUNRGBD'):
        data_path = '/home/dongwonshin/Desktop/Datasets/SUNRGBD/SUNRGBD(light)/'
        t_loader = SUNRGBDLoader(data_path, is_transform=True)
        v_loader = SUNRGBDLoader(data_path, is_transform=True, split='val')

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=16, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=16)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup Model
    print(model_name)
    model = get_model(model_name, n_classes)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    model.cuda()

    for i, (images, depths, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        images = Variable(images.cuda(), volatile=True)
        depths = Variable(depths.cuda(), volatile=True)
        #labels = Variable(labels.cuda(), volatile=True)

        if (model_name == 'fcn8s'):
            outputs = model(images)
        else:
            outputs = model(images, depths)

        pred = outputs.data.max(1)[1].cpu().numpy()

        #gt = labels.data.cpu().numpy()
        gt = labels.numpy()

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            # print('Inference time (iter {0:5d}): {1:3.5f} fps'.format(i+1, pred.shape[0]/elapsed_time))
            sys.stdout.write('.')
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--eval_flip', dest='eval_flip', action='store_true', 
                        help='Enable evaluation with flipped image | True by default')
    parser.add_argument('--no-eval_flip', dest='eval_flip', action='store_false', 
                        help='Disable evaluation with flipped image | True by default')
    parser.set_defaults(eval_flip=True)

    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val', 
                        help='Split of dataset to test on')

    parser.add_argument('--measure_time', dest='measure_time', action='store_true', 
                        help='Enable evaluation with time (fps) measurement | True by default')
    parser.add_argument('--no-measure_time', dest='measure_time', action='store_false', 
                        help='Disable evaluation with time (fps) measurement | True by default')
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()
    validate(args)
