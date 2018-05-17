import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.fcn_with_maskedconv import *
from ptsemseg.models.fcn8s_with_rgbd import *
from ptsemseg.models.fcn8s_rgbd_renet import *
from ptsemseg.models.fcn8s_rgbd_renet_maskedconv import *
from ptsemseg.models.mynetwork20180516 import *
from ptsemseg.models.mynetwork20180517 import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *


def get_model(name, n_classes, version=None):
    model = _get_model_instance(name)

    if name in ['frrnA', 'frrnB']:
        model = model(n_classes, model_type=name[-1])

    elif name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
        
    elif name == 'fcn(masked)':
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    
    elif name == 'fcn8s_with_rgbd':
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
        
    elif name == 'fcn8s_rgbd_renet':
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
        
    elif name == 'fcn8s_rgbd_renet_maskedconv':
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    
    elif name in ['mynetwork20180516', 'mynetwork20180517']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
        
    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'unet':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)

    elif name == 'pspnet':
        model = model(n_classes=n_classes, version=version)

    elif name == 'icnet':
        model = model(n_classes=n_classes, with_bn=False, version=version)
    elif name == 'icnetBN':
        model = model(n_classes=n_classes, with_bn=True, version=version)

    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'fcn32s': fcn32s,
            'fcn8s': fcn8s,
            'fcn16s': fcn16s,
            'fcn(masked)' : fcn_with_maskedconv,
            'fcn8s_with_rgbd' : fcn8s_with_rgbd,
            'fcn8s_rgbd_renet' : fcn8s_rgbd_renet,
            'fcn8s_rgbd_renet_maskedconv' : fcn8s_rgbd_renet_maskedconv,
            'mynetwork20180516' : mynetwork20180516,
            'mynetwork20180517' : mynetwork20180517,
            'unet': unet,
            'segnet': segnet,
            'pspnet': pspnet,
			'icnet': icnet,
			'icnetBN': icnet,
            'linknet': linknet,
            'frrnA': frrn,
            'frrnB': frrn,
        }[name]
    except:
        print('Model {} not available'.format(name))
