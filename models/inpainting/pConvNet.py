"""
References:
https://github.com/NVIDIA/partialconv

typing sources with reading paper for tutorial...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PConv2d(nn.Module):
    # Partial Convolution 2d layer
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PConv2d, self).__init__()
        self.conv2d_feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                        bias=bias)
        self.conv2d_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                     bias=False)

    def forward(self, inputs, mask):
        # X ⊙ M
        #         print('inputs shape : ',inputs.shape)
        #         print('mask shape : ',mask.shape)
        valid_feature = inputs * mask
        out_feature = self.conv2d_feature(valid_feature)

        # get bias for 1/sum(M)
        if self.conv2d_feature.bias is not None:
            out_bias = self.conv2d_feature.bias.view(1, -1, 1, 1).expand_as(out_feature)
        else:
            out_bias = torch.zeros_like(out_feature)

        #         print('out_bias shape : ',out_bias.shape)

        # mask does not require gradients
        with torch.no_grad():
            out_masking = self.conv2d_mask(mask)

        #         print('out_masking shape : ',out_masking.shape)
        invalid_mask = (out_masking == 0)
        sum_mask = out_masking.masked_fill_(invalid_mask, 1.0)
        #         print('sum_mask shape : ',sum_mask.shape)

        out_feature = (out_feature - out_bias) / sum_mask + out_bias
        out_feature = out_feature.masked_fill_(invalid_mask, 0.0)

        out_mask = torch.ones_like(out_feature)
        out_mask = out_mask.masked_fill_(invalid_mask, 0.0)

        return out_feature, out_mask


class PConv2dBlock(nn.Module):
    # Partial Convolution 2d layer with activation, batch normalization
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 bn=True, activation='relu', slope_leaky=0.2):
        super(PConv2dBlock, self).__init__()

        self.pconv2d = PConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=slope_leaky)

    def forward(self, inputs, mask):
        out_feature, out_mask = self.pconv2d(inputs, mask)

        if hasattr(self, 'bn'):
            out_feature = self.bn(out_feature)
        if hasattr(self, 'activation'):
            out_feature = self.activation(out_feature)

        return out_feature, out_mask


class PConvNet(nn.Module):
    def __init__(self, input_channels=3, layer_size=7):
        super(PConvNet, self).__init__()

        self.layer_size = layer_size
        self.freeze_enc_bn = False

        # Encoding
        # Assume input shape : (batch_size, input_channels, 512, 512)
        self.enc_1 = PConv2dBlock(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, bn=False)
        # out shape : (batch_size, 64, 256, 256)
        self.enc_2 = PConv2dBlock(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        # out shape : (batch_size, 128, 128, 128)
        self.enc_3 = PConv2dBlock(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        # out shape : (batch_size, 256, 64, 64)
        self.enc_4 = PConv2dBlock(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        # out shape : (batch_size, 512, 32, 32)
        for i in range(4, 7):
            layer_name = 'enc_{}'.format(i + 1)
            setattr(self, layer_name, PConv2dBlock(512, 512, kernel_size=3, stride=2, padding=1, bias=False))
        # out shape : (batch_size, 512, 16, 16)
        # out shape : (batch_size, 512, 8, 8)
        # out shape : (batch_size, 512, 4, 4)

        # Decoding
        # Channelwise concatenation with each output of encoding layers
        for i in range(4, 7):
            layer_name = 'dec_{}'.format(i + 1)
            setattr(self, layer_name, PConv2dBlock(512 + 512, 512, kernel_size=3, stride=1, padding=1, bias=False,
                                                   activation='LeakyReLU'))
        self.dec_4 = PConv2dBlock(512 + 256, 256, kernel_size=3, stride=1, padding=1, bias=False,
                                  activation='LeakyReLU')
        self.dec_3 = PConv2dBlock(256 + 128, 128, kernel_size=3, stride=1, padding=1, bias=False,
                                  activation='LeakyReLU')
        self.dec_2 = PConv2dBlock(128 + 64, 64, kernel_size=3, stride=1, padding=1, bias=False, activation='LeakyReLU')
        self.dec_1 = PConv2dBlock(64 + input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=True,
                                  bn=False, activation=None)

    def forward(self, inputs, mask, upsampling_mode='nearest'):
        layer_dict = dict()
        layer_mask_dict = dict()

        layer_dict['enc_0'], layer_mask_dict['enc_0'] = inputs, mask

        enc_key_prev = 'enc_0'

        # Encoding
        for i in range(0, 7):
            enc_key = 'enc_{}'.format(i + 1)
            layer_dict[enc_key], layer_mask_dict[enc_key] = getattr(self, enc_key)(
                layer_dict[enc_key_prev], layer_mask_dict[enc_key_prev])

            enc_key_prev = enc_key

        dec, dec_mask = layer_dict[enc_key], layer_mask_dict[enc_key]

        # Decoding
        for i in range(7, 0, -1):
            enc_key = 'enc_{}'.format(i - 1)
            dec_key = 'dec_{}'.format(i)

            dec = F.upsample(dec, scale_factor=2, mode=upsampling_mode)
            dec_mask = F.upsample(dec_mask, scale_factor=2, mode='nearest')

            dec = torch.cat([dec, layer_dict[enc_key]], dim=1)
            dec_mask = torch.cat([dec_mask, layer_mask_dict[enc_key]], dim=1)

            dec, dec_mask = getattr(self, dec_key)(dec, dec_mask)

        return dec, dec_mask

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = vgg16.features[:5]
        self.enc_2 = vgg16.features[5:10]
        self.enc_3 = vgg16.features[10:17]

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PConvLoss(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super(PConvLoss, self).__init__()

        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, inputs, mask, output, gt):
        loss_dict = dict()
        output_comp = mask * inputs + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp] * 3, 1))
            feat_output = self.extractor(torch.cat([output] * 3, 1))
            feat_gt = self.extractor(torch.cat([gt] * 3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['perceptual'] = 0.0
        for i in range(3):
            loss_dict['perceptual'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['perceptual'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict