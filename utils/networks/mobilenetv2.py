import math
import torch
from torch import nn
from ..graph import *


__all__ = ['MobileNetV2', 'mobilenetv2']


def my_round(x):
    if x == 22.5:
        return 23
    elif x == 16.5:
        return 17
    elif x == 10.5:
        return 11
    elif x == 4.5:
        return 5
    else:
        return round(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, channel_ratio=1.0):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(my_round(inp*channel_ratio), my_round(hidden_dim*channel_ratio), kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(my_round(hidden_dim*channel_ratio), my_round(hidden_dim*channel_ratio), stride=stride, groups=my_round(hidden_dim*channel_ratio)),
            # pw-linear
            nn.Conv2d(my_round(hidden_dim*channel_ratio), my_round(oup*channel_ratio), 1, 1, 0, bias=False),
            nn.BatchNorm2d(my_round(oup*channel_ratio)),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def parse_conv_bn_relu_to_graph(module, prefix, head_node, graph):
    """
    0. conv
    1. bn
    2. relu
    """
    info = get_module_info(module[0])
    curr_node = Node(prefix+'.0', 'conv', info, graph)
    head_node.add_link_to(curr_node)
    head_node = curr_node

    info = get_module_info(module[1])
    curr_node = Node(prefix+'.1', 'bn', info, graph)
    head_node.add_link_to(curr_node)
    head_node = curr_node

    inp = module[0].in_channels
    oup = module[0].out_channels
    stride = module[0].stride
    return head_node, (inp, oup, stride)


def parse_inverted_residual_to_graph(module, prefix, head_node, graph):
    """
    0. ConvBNReLU ( except for prefix == features.1 )
    1. ConvBNReLU
    2. conv
    3. bn
    """
    root_node = head_node
    block_inp, block_oup = -1, -1
    block_stride = (-1, -1)
    for index, submodule in module.conv.named_children():

        index = int(index)
        module_name = prefix + '.conv.{}'.format(index)
        if isinstance(submodule, ConvBNReLU):
            head_node, block_info = parse_conv_bn_relu_to_graph(
                    submodule, module_name, head_node, graph)

            if index == 0: block_inp = block_info[0]
            if index == 1: block_stride = block_info[2]
        else:
            module_type = 'conv' if isinstance(submodule, nn.Conv2d) else 'bn'
            info = get_module_info(submodule)
            curr_node = Node(module_name, module_type, info, graph)
            head_node.add_link_to(curr_node)
            head_node = curr_node

            if module_type == 'bn':
                block_oup = submodule.num_features

    if block_inp ==  block_oup and block_stride == (1, 1):
        curr_node = AddNode(graph)
        root_node.add_link_to(curr_node)
        head_node.add_link_to(curr_node)
        head_node = curr_node
    return head_node


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, channel_ratio=1.0, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel  = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, my_round(input_channel*channel_ratio), stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, channel_ratio=channel_ratio))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(my_round(input_channel*channel_ratio), my_round(self.last_channel*channel_ratio), kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(my_round(self.last_channel*channel_ratio), num_classes),
        )
        self._initialize_weights()
        return

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def build_graph(self):
        # Construct the graph
        info = get_module_info(self.features[0][0])
        first_conv_node = Node('features.0.0', 'conv', info)
        graph = Graph(start_node=first_conv_node, name='MobileNetV2')

        info = get_module_info(self.features[0][1])
        first_bn_node = Node('features.0.1', 'bn', info, graph)
        first_conv_node.add_link_to(first_bn_node)

        head_node = first_bn_node
        for block_idx, module in self.features.named_children():
            block_idx = int(block_idx)
            if block_idx == 0: continue

            module_name = 'features.{}'.format(block_idx)
            if isinstance(module, InvertedResidual):
                head_node = parse_inverted_residual_to_graph(
                        module, module_name, head_node, graph)
            else:
                head_node, _ = parse_conv_bn_relu_to_graph(
                        module, module_name, head_node, graph)

        info = get_module_info(self.classifier[1])
        fc_node = Node('classifier.1', 'fc', info, graph)
        head_node.add_link_to(fc_node)
        return graph

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return


def mobilenetv2(num_classes=1000, channel_ratio=1.0):
    model = MobileNetV2(num_classes=num_classes, channel_ratio=channel_ratio)
    return model
