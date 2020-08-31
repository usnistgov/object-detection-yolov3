# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys

if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import torch
import torch.nn

from yolo_layer import YOLOLayer


def conv_layer(fc_in, fc_out, kernel, stride=1):
    """
        Add a conv2d / batchnorm / leaky ReLU block.
        Args:
            fc_in (int): number of input channels of the convolution layer.
            fc_out (int): number of output channels of the convolution layer.
            kernel (int): kernel size of the convolution layer.
            stride (int): stride of the convolution layer.
        Returns:
            stage (Sequential) : Sequential layers composing a convolution block.
        """
    stage = torch.nn.Sequential()
    pad = (kernel - 1) // 2
    stage.add_module('conv', torch.nn.Conv2d(in_channels=fc_in,
                                             out_channels=fc_out, kernel_size=kernel, stride=stride,
                                             padding=pad, bias=False))
    stage.add_module('batch_norm', torch.nn.BatchNorm2d(fc_out))
    stage.add_module('leaky', torch.nn.LeakyReLU(YoloV3.LEAKY_VALUE))
    return stage


class res_block(torch.nn.Module):
    """
    Sequential residual blocks each of which consists of two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nb_reps (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nb_reps=1, kernel=3, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = torch.nn.ModuleList()
        for i in range(nb_reps):
            resblock_one = torch.nn.ModuleList()
            resblock_one.append(conv_layer(fc_in=int(ch), fc_out=int(ch / 2), kernel=1, stride=1))
            resblock_one.append(conv_layer(fc_in=int(ch / 2), fc_out=int(ch), kernel=kernel, stride=1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x




def build_net_dict(config_model, ignore_thres, number_channels, image_size):
    # ---- DarkNet53 ----
    mdict = torch.nn.ModuleDict()

    # line 25 of https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    mdict['dn0_conv'] = conv_layer(fc_in=number_channels, fc_out=int(YoloV3.FILTER_COUNT / 32), kernel=YoloV3.KERNEL_SIZE, stride=1)

    # line 35
    mdict['dn1_conv'] = conv_layer(fc_in=int(YoloV3.FILTER_COUNT / 32), fc_out=int(YoloV3.FILTER_COUNT / 16), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

    # build 1x copies of metablock 1
    mdict['dn2_rb1'] = res_block(ch=int(YoloV3.FILTER_COUNT / 16), nb_reps=1, kernel=YoloV3.KERNEL_SIZE)

    # line 65
    mdict['dn3_rb1_to_rb2_conv'] = conv_layer(fc_in=int(YoloV3.FILTER_COUNT / 16), fc_out=int(YoloV3.FILTER_COUNT / 8), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

    # build 2x copies of metablock 2
    mdict['dn4_rb2'] = res_block(ch=int(YoloV3.FILTER_COUNT / 8), nb_reps=2, kernel=YoloV3.KERNEL_SIZE)

    # line 115
    mdict['dn5_rb2_to_rb3_conv'] = conv_layer(fc_in=int(YoloV3.FILTER_COUNT / 8), fc_out=int(YoloV3.FILTER_COUNT / 4), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

    # build Nx copies of metablock 3
    fm_4x_fc = int(YoloV3.FILTER_COUNT / 4)
    mdict['dn6_route0_rb3'] = res_block(ch=fm_4x_fc, nb_reps=YoloV3.BLOCK_COUNT, kernel=YoloV3.KERNEL_SIZE)

    # line 286
    mdict['dn7_rb3_to_rb4_conv'] = conv_layer(fc_in=int(YoloV3.FILTER_COUNT / 4), fc_out=int(YoloV3.FILTER_COUNT / 2), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

    # build Nx copies of metablock 4
    fm_2x_fc = int(YoloV3.FILTER_COUNT / 2)
    mdict['dn8_route1_rb4'] = res_block(ch=fm_2x_fc, nb_reps=YoloV3.BLOCK_COUNT, kernel=YoloV3.KERNEL_SIZE)

    # line 461
    mdict['dn9_rb4_to_rb5_conv'] = conv_layer(fc_in=int(YoloV3.FILTER_COUNT / 2), fc_out=int(YoloV3.FILTER_COUNT), kernel=YoloV3.KERNEL_SIZE, stride=2)  # downsample 50% using stride=2

    # build (N/2)x copies of metablock 5
    fm_1x_fc = int(YoloV3.FILTER_COUNT)
    mdict['dn10_route2_rb5'] = res_block(ch=fm_1x_fc, nb_reps=int(YoloV3.BLOCK_COUNT / 2), kernel=YoloV3.KERNEL_SIZE)

    # output_layer_name5 is tensor with shape <batch_size>, 1024, <img_size>/32, <img_size>/32
    # downsample_factor = 32

    # ---- Yolov3 ----
    mdict['y11_rb'] = res_block(ch=int(fm_1x_fc), nb_reps=2, kernel=3, shortcut=False)
    mdict['y12_conv'] = conv_layer(fc_in=int(fm_1x_fc), fc_out=int(fm_1x_fc / 2), kernel=1, stride=1)
    # 1st yolo branch
    mdict['y13_conv'] = conv_layer(fc_in=int(fm_1x_fc / 2), fc_out=int(fm_1x_fc), kernel=YoloV3.KERNEL_SIZE, stride=1)
    mdict['y14_yolo_layer0'] = YOLOLayer(config_model, layer_nb=0, in_ch=int(fm_1x_fc), ignore_thres=ignore_thres)

    mdict['y15_conv'] = conv_layer(fc_in=int(fm_2x_fc), fc_out=int(fm_2x_fc / 2), kernel=1, stride=1)
    # mdict['y16_upsample'] = torch.nn.Upsample(scale_factor=2, mode='nearest')
    tgt_h = int(image_size[0] / 16)
    tgt_w = int(image_size[0] / 16)
    mdict['y16_upsample'] = torch.nn.Upsample(size=(tgt_h, tgt_w), mode='nearest')

    # input is concat of y12_conv and dn8_route1_rb4
    fc = int(fm_1x_fc / 2) + int(fm_2x_fc / 2)
    mdict['y17_conv'] = conv_layer(fc_in=int(fc), fc_out=int(fm_2x_fc / 2), kernel=1, stride=1)
    mdict['y18_conv'] = conv_layer(fc_in=int(fm_2x_fc / 2), fc_out=int(fm_2x_fc), kernel=YoloV3.KERNEL_SIZE, stride=1)
    mdict['y19_rb'] = res_block(ch=int(fm_2x_fc), nb_reps=1, kernel=YoloV3.KERNEL_SIZE, shortcut=False)
    mdict['y20_conv'] = conv_layer(fc_in=int(fm_2x_fc), fc_out=int(fm_2x_fc / 2), kernel=1, stride=1)
    # 2nd yolo branch
    mdict['y21_conv'] = conv_layer(fc_in=int(fm_2x_fc / 2), fc_out=int(fm_2x_fc), kernel=YoloV3.KERNEL_SIZE, stride=1)
    mdict['y22_yolo_layer1'] = YOLOLayer(config_model, layer_nb=1, in_ch=int(fm_2x_fc), ignore_thres=ignore_thres)

    mdict['y23_conv'] = conv_layer(fc_in=int(fm_4x_fc), fc_out=int(fm_4x_fc / 2), kernel=1, stride=1)
    # mdict['y24_upsample'] = torch.nn.Upsample(scale_factor=2, mode='nearest')
    tgt_h = int(image_size[0] / 8)
    tgt_w = int(image_size[0] / 8)
    mdict['y24_upsample'] = torch.nn.Upsample(size=(tgt_h, tgt_w), mode='nearest')

    # input is concat of y20_conv and dn6_route0_rb3
    fc = int(fm_2x_fc / 2) + int(fm_4x_fc / 2)
    mdict['y25_conv'] = conv_layer(fc_in=int(fc), fc_out=int(fm_4x_fc / 2), kernel=1, stride=1)
    mdict['y26_conv'] = conv_layer(fc_in=int(fm_4x_fc / 2), fc_out=int(fm_4x_fc), kernel=YoloV3.KERNEL_SIZE, stride=1)
    mdict['y27_rb'] = res_block(ch=int(fm_4x_fc), nb_reps=2, kernel=YoloV3.KERNEL_SIZE, shortcut=False)
    # 3rd yolo branch
    mdict['y28_yolo_layer2'] = YOLOLayer(config_model, layer_nb=2, in_ch=int(fm_4x_fc), ignore_thres=ignore_thres)

    return mdict


class YoloV3(torch.nn.Module):
    # Constants controlling the network
    BLOCK_COUNT = 8
    FILTER_COUNT = 1024
    KERNEL_SIZE = 3
    NETWORK_DOWNSAMPLE_FACTOR = 32
    LEAKY_VALUE = 0.1

    # # [batch_size, 5376, 5 + NUMBER_CLASSES]
    # # last column contents: [x_ul, y_ul, x_lr, y_lr, objectness_logit, class_probability]
    # # class label is a column vector containing the index of the class (in floating point representation to allow concatenation with the floating point boxes)

    def __init__(self, config_model, ignore_thres=0.5):
        super(YoloV3, self).__init__()

        self.image_size = config_model['image_size']  # [h, w, c]
        self.ignore_thres = ignore_thres
        self.number_classes = config_model['number_classes']
        self.number_channels = self.image_size[2]  # image size is [h, w, c]
        self.mdict = build_net_dict(config_model, self.ignore_thres, self.number_channels, self.image_size)

    def forward(self, x, targets=None):
        """
        Forward path of YoloV3.
        Args:
           x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
               where N, C are batchsize and num. of channels.
           targets (torch.Tensor) : label array whose shape is :math:`(?, ?, ?, ?)`
        Returns:
           training:
               output (torch.Tensor): loss tensor for backpropagation.
           test:
               output (torch.Tensor): concatenated detection results.
        """

        train = targets is not None
        output = []

        x = self.mdict['dn0_conv'](x)
        x = self.mdict['dn1_conv'](x)
        x = self.mdict['dn2_rb1'](x)
        x = self.mdict['dn3_rb1_to_rb2_conv'](x)
        x = self.mdict['dn4_rb2'](x)
        x = self.mdict['dn5_rb2_to_rb3_conv'](x)
        x = self.mdict['dn6_route0_rb3'](x)
        feature_map_4x_res = x
        x = self.mdict['dn7_rb3_to_rb4_conv'](x)
        x = self.mdict['dn8_route1_rb4'](x)
        feature_map_2x_res = x
        x = self.mdict['dn9_rb4_to_rb5_conv'](x)
        x = self.mdict['dn10_route2_rb5'](x)
        # feature_map_1x_res = x

        x = self.mdict['y11_rb'](x)
        x = self.mdict['y12_conv'](x)
        route = x
        x = self.mdict['y13_conv'](x)
        if train:
            x = self.mdict['y14_yolo_layer0'](x, targets)
        else:
            x = self.mdict['y14_yolo_layer0'](x)
        output.append(x)

        x = self.mdict['y15_conv'](route)
        x = self.mdict['y16_upsample'](x)
        # cat dim 1 because NCHW
        x = torch.cat((x, feature_map_2x_res), dim=1)

        x = self.mdict['y17_conv'](x)
        x = self.mdict['y18_conv'](x)
        x = self.mdict['y19_rb'](x)
        x = self.mdict['y20_conv'](x)
        route = x
        x = self.mdict['y21_conv'](x)
        if train:
            x = self.mdict['y22_yolo_layer1'](x, targets)
        else:
            x = self.mdict['y22_yolo_layer1'](x)
        output.append(x)

        x = self.mdict['y23_conv'](route)
        x = self.mdict['y24_upsample'](x)
        # cat dim 1 because NCHW
        x = torch.cat((x, feature_map_4x_res), dim=1)

        x = self.mdict['y25_conv'](x)
        x = self.mdict['y26_conv'](x)
        x = self.mdict['y27_rb'](x)
        if train:
            x = self.mdict['y28_yolo_layer2'](x, targets)
        else:
            x = self.mdict['y28_yolo_layer2'](x)
        output.append(x)

        if train:
            output = torch.cat(output, dim=0)
            output = torch.sum(output, dim=0, keepdim=True)
            return output
        else:
            # output = torch.cat(output, dim=1)
            return output


