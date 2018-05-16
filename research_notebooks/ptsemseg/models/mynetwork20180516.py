import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ReNet(nn.Module):

    def __init__(self, n_input, n_units, patch_size=(1, 1), usegpu=True):
        super(ReNet, self).__init__()

        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])

        assert self.patch_size_height >= 1
        assert self.patch_size_width >= 1

        self.tiling = False if ((self.patch_size_height == 1) and (self.patch_size_width == 1)) else True

        self.rnn_hor = nn.GRU(n_input * self.patch_size_height * self.patch_size_width, n_units,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_ver = nn.GRU(n_units * 2, n_units, num_layers=1, batch_first=True, bidirectional=True)

    def tile(self, x):

        n_height_padding = self.patch_size_height - x.size(2) % self.patch_size_height
        n_width_padding = self.patch_size_width - x.size(3) % self.patch_size_width

        n_top_padding = n_height_padding / 2
        n_bottom_padding = n_height_padding - n_top_padding

        n_left_padding = n_width_padding / 2
        n_right_padding = n_width_padding - n_left_padding

        x = F.pad(x, (n_left_padding, n_right_padding, n_top_padding, n_bottom_padding))

        b, n_filters, n_height, n_width = x.size()

        assert n_height % self.patch_size_height == 0
        assert n_width % self.patch_size_width == 0

        new_height = n_height / self.patch_size_height
        new_width = n_width / self.patch_size_width

        x = x.view(b, n_filters, new_height, self.patch_size_height, new_width, self.patch_size_width)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(b, new_height, new_width, self.patch_size_height * self.patch_size_width * n_filters)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        return x

    def rnn_forward(self, x, hor_or_ver):

        assert hor_or_ver in ['hor', 'ver']

        b, n_height, n_width, n_filters = x.size()

        x = x.view(b * n_height, n_width, n_filters)
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        else:
            x, _ = self.rnn_ver(x)
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)

        return x

    def forward(self, x):

                                       #b, nf, h, w
        if self.tiling:
            x = self.tile(x)           #b, nf, h, w
        x = x.permute(0, 2, 3, 1)      #b, h, w, nf
        x = x.contiguous()
        x = self.rnn_forward(x, 'hor') #b, h, w, nf
        x = x.permute(0, 2, 1, 3)      #b, w, h, nf
        x = x.contiguous()
        x = self.rnn_forward(x, 'ver') #b, w, h, nf
        x = x.permute(0, 2, 1, 3)      #b, h, w, nf
        x = x.contiguous()
        x = x.permute(0, 3, 1, 2)      #b, nf, h, w
        x = x.contiguous()

        return x
    

# FCN 8s
class mynetwork20180516(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=False):
        super(mynetwork20180516, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)
        
        self.depth_conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),)
        
        fm = 512
        self.masked_conv = nn.Sequential(
            MaskedConv2d('A', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
            nn.Conv2d(fm, 512, 1)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),)

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, color, depth):
        conv1 = self.conv_block1(color)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        masked_conv = self.masked_conv(conv5)
        
        score = self.classifier(masked_conv)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        
        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4        
        score = F.upsample_bilinear(score, score_pool3.size()[2:])
        score += score_pool3
        
        depth_conv1 = self.depth_conv_block1(depth)
        depth_conv2 = self.conv_block2(depth_conv1)
        depth_conv3 = self.conv_block3(depth_conv2)
        depth_conv4 = self.conv_block4(depth_conv3)
        depth_conv5 = self.conv_block5(depth_conv4)
        depth_masked_conv = self.masked_conv(depth_conv5)
        
        depth_score = self.classifier(depth_masked_conv)
        depth_score_pool4 = self.score_pool4(depth_conv4)
        depth_score_pool3 = self.score_pool3(depth_conv3)

        depth_score = F.upsample_bilinear(depth_score, depth_score_pool4.size()[2:])
        depth_score += depth_score_pool4        
        depth_score = F.upsample_bilinear(depth_score, depth_score_pool3.size()[2:])
        depth_score += depth_score_pool3
        
        score += depth_score
        
        out = F.upsample_bilinear(score, color.size()[2:])

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
