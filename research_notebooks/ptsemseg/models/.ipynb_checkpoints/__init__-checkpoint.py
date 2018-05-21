import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.fcn_with_maskedconv import *
from ptsemseg.models.fcn8s_with_rgbd import *
from ptsemseg.models.fcn8s_rgbd_renet import *
from ptsemseg.models.fcn8s_rgbd_renet_maskedconv import *
from ptsemseg.models.mynetwork20180516 import *
from ptsemseg.models.fcn_hha import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *

from ptsemseg.models.FCN_RGB import *
from ptsemseg.models.FCN_RGB_mask import *
from ptsemseg.models.FCN_RGB_mask2 import *
from ptsemseg.models.FCN_RGB_renet import *
from ptsemseg.models.FCN_RGBD import *
from ptsemseg.models.FCN_RGBD_mask import *
from ptsemseg.models.FCN_RGBD_mask2 import *
from ptsemseg.models.FCN_RGBD_renet import *

from ptsemseg.models.MaskedSegnet import *
from ptsemseg.models.RecurrentSegnet import *
from ptsemseg.models.GlocalContextNet import *

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
    
    elif name == 'mynetwork20180516':
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    
    elif name == 'fcn_hha':
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    
    elif name in ['FCN_RGB', 'FCN_RGB_mask', 'FCN_RGB_mask2', 'FCN_RGB_renet',
                  'FCN_RGBD', 'FCN_RGBD_mask', 'FCN_RGBD_mask2' ,'FCN_RGBD_renet']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
        
    elif name in ['segnet', 'GlocalContextNet']:
        model = model(n_classes=n_classes, is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    
    elif name in ['MaskedSegnet','RecurrentSegnet']:
        model = model(n_classes=n_classes, is_unpooling=True)
        
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
            'fcn_hha' : fcn_hha,
            'FCN_RGB' : FCN_RGB,
            'FCN_RGB_mask' : FCN_RGB_mask,
            'FCN_RGB_mask2' : FCN_RGB_mask2,
            'FCN_RGB_renet' : FCN_RGB_renet,
            'FCN_RGBD' : FCN_RGBD,
            'FCN_RGBD_mask' : FCN_RGBD_mask,
            'FCN_RGBD_mask2' : FCN_RGBD_mask2,
            'FCN_RGBD_renet' : FCN_RGBD_renet,
            'unet': unet,
            'segnet': segnet,
            'MaskedSegnet': MaskedSegnet,
            'RecurrentSegnet': RecurrentSegnet,
            'GlocalContextNet': GlocalContextNet,
            'pspnet': pspnet,
            'icnet': icnet,
            'icnetBN': icnet,
            'linknet': linknet,
            'frrnA': frrn,
            'frrnB': frrn,
        }[name]
    except:
        print('Model {} not available'.format(name))
