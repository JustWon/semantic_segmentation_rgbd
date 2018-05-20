import os
import argparse

import visdom
from tqdm import tqdm

from torch.autograd import Variable

from ptsemseg.metrics import runningScore
from ptsemseg.models import get_model
from ptsemseg.loss import *

from NYUDv2Loader import *

class trainer:
        
    def _parser_setting(self):

        parser = argparse.ArgumentParser(description='Hyperparams')
        parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', help='Architecture to use [\'fcn8s, unet, segnet etc\']')
        parser.add_argument('--img_rows', nargs='?', type=int, default=256, help='Height of the input image')
        parser.add_argument('--img_cols', nargs='?', type=int, default=256, help='Width of the input image')

        parser.add_argument('--img_norm', dest='img_norm', action='store_true', help='Enable input image scales normalization [0, 1] | True by default')
        parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', help='Disable input image scales normalization [0, 1] | True by default')
        parser.set_defaults(img_norm=True)

        parser.add_argument('--n_epoch', nargs='?', type=int, default=10, help='# of the epochs')
        parser.add_argument('--batch_size', nargs='?', type=int, default=1, help='Batch Size')
        parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, help='Learning Rate')
        parser.add_argument('--feature_scale', nargs='?', type=int, default=1, help='Divider for # of features to use')
        parser.add_argument('--resume', nargs='?', type=str, default=None, help='Path to previous saved model to restart from')

        parser.add_argument('--visdom', dest='visdom', action='store_true', help='Enable visualization(s) on visdom | False by default')
        parser.add_argument('--no-visdom', dest='visdom', action='store_false', help='Disable visualization(s) on visdom | False by default')
        parser.set_defaults(visdom=False)

        parser.add_argument('--gpu_idx', type=str, default=0)
        parser.add_argument('--dataset', type=str, default=0)
        parser.add_argument('--input_type', type=str, default=0)

        return parser

    def __init__(self, arg_str):
        self.parser = self._parser_setting()
        self.args = self.parser.parse_args(arg_str.split(' '))
        
        self.data_path = '/home/dongwonshin/Desktop/Datasets/NYUDv2/'
    def __del__(self):
        torch.cuda.empty_cache()

    def model_init(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_idx
        
        # Setup Dataloader
        self.t_loader = NYUDv2Loader(self.data_path, is_transform=True)
        self.v_loader = NYUDv2Loader(self.data_path, is_transform=True, split='val')

        self.n_classes = self.t_loader.n_classes
        self.trainloader = data.DataLoader(self.t_loader, batch_size=self.args.batch_size, num_workers=16, shuffle=True)
        self.valloader = data.DataLoader(self.v_loader, batch_size=self.args.batch_size, num_workers=16)

        # Setup Metrics
        self.running_metrics = runningScore(self.n_classes)

        # Setup visdom for visualization
        if self.args.visdom:
            self.vis = visdom.Visdom()
            vis_title = '%s_%s' % (self.args.arch, self.args.dataset)
            self.loss_window = self.vis.line(X=torch.zeros((1,)).cpu(),
                                        Y=torch.zeros((1)).cpu(),
                                        opts=dict(xlabel='minibatches', ylabel='Loss', title=vis_title, legend=['Loss']))

        # Setup Model
        self.model = get_model(self.args.arch, self.n_classes)

        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model.cuda()

        # Check if model has custom optimizer / loss
        if hasattr(self.model.module, 'optimizer'):
            self.optimizer = self.model.module.optimizer
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.l_rate, momentum=0.99, weight_decay=5e-4)

        if hasattr(self.model.module, 'loss'):
            print('Using custom loss')
            self.loss_fn = self.model.module.loss
        else:
            self.loss_fn = cross_entropy2d

        if self.args.resume is not None:                                         
            if os.path.isfile(self.args.resume):
                print("Loading model and optimizer from checkpoint '{}'".format(self.args.resume))
                self.checkpoint = torch.load(self.args.resume)
                self.model.load_state_dict(self.checkpoint['model_state'])
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state'])
                print("Loaded checkpoint '{}' (epoch {})"                    
                      .format(self.args.resume, self.checkpoint['epoch']))
            else:
                print("No checkpoint found at '{}'".format(self.args.resume)) 


    def training(self):
        best_iou = -100.0 
        x_pos = 0
        for epoch in range(self.args.n_epoch):

            # train
            self.model.train()
            for i, (color_imgs, depth_imgs, label_imgs) in enumerate(self.trainloader):
                color_imgs = Variable(color_imgs.cuda())
                label_imgs = Variable(label_imgs.cuda())
                
                if (self.args.input_type == 'RGBD'):
                    depth_imgs = Variable(depth_imgs.cuda())

                self.optimizer.zero_grad()
                
                if (self.args.input_type == 'RGB'):
                    outputs = self.model(color_imgs)
                elif (self.args.input_type == 'RGBD'):
                    outputs = self.model(color_imgs, depth_imgs)

                loss = self.loss_fn(input=outputs, target=label_imgs)

                loss.backward()
                self.optimizer.step()
                
                if self.args.visdom:
                    self.vis.line(
                        X=torch.ones((1, 1)).cpu() * x_pos,
                        Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                        win=self.loss_window,
                        update='append')
                    x_pos += 1

                if (i+1) % 100 == 0:
                    print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, self.args.n_epoch, loss.data[0]))

            # eval
            self.model.eval()
            for i_val, (color_images_val, depth_images_val, label_images_val) in tqdm(enumerate(self.valloader)):
                color_images_val = Variable(color_images_val.cuda(), volatile=True)
                label_images_val = Variable(label_images_val.cuda(), volatile=True)
                
                if (self.args.input_type == 'RGBD'):
                    depth_images_val = Variable(depth_images_val.cuda(), volatile=True)

                if (self.args.input_type == 'RGB'):
                    outputs = self.model(color_images_val)
                elif (self.args.input_type == 'RGBD'):
                    outputs = self.model(color_images_val, depth_images_val)
                    
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = label_images_val.data.cpu().numpy()
                self.running_metrics.update(gt, pred)

            score, class_iou = self.running_metrics.get_scores()
            for k, v in score.items():
                print(k, v)
            self.running_metrics.reset()

            # model save
            if score['Mean IoU : \t'] >= best_iou:
                best_iou = score['Mean IoU : \t']
                state = {'epoch': epoch+1,
                         'model_state': self.model.state_dict(),
                         'optimizer_state' : self.optimizer.state_dict(),}
                torch.save(state, "../model_weights/{}_{}_best_model.pkl".format(self.args.arch, self.args.dataset))