import math
import torch
import torch.nn as nn
from ..graph import *


__all__ = ['mobilenetv1']


def parse_dw_conv_to_graph(module, block_idx, head_node, graph):
    """
    0. conv
    1. bn
    2. relu
    3. conv
    4. bn
    5. relu
    """
    for layer_idx, submodule in module.named_children():

        layer_idx = int(layer_idx)
        module_name = 'model.{}.{}'.format(block_idx, layer_idx)

        if layer_idx in [0, 3]:
            module_type = 'conv'
            info = get_module_info(submodule)
        elif layer_idx in [1, 4]:
            module_type = 'bn'
            info = get_module_info(submodule)

        if layer_idx in [2, 5]: continue

        curr_node = Node(module_name, module_type, info, graph)
        head_node.add_link_to(curr_node)
        head_node = curr_node
    return head_node



class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, channel_ratio=1.0):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(                      3,   round(channel_ratio*32),   2),
            conv_dw(round(channel_ratio*32),   round(channel_ratio*64),   1),
            conv_dw(round(channel_ratio*64),   round(channel_ratio*128),  2),
            conv_dw(round(channel_ratio*128),  round(channel_ratio*128),  1),
            conv_dw(round(channel_ratio*128),  round(channel_ratio*256),  2),
            conv_dw(round(channel_ratio*256),  round(channel_ratio*256),  1),
            conv_dw(round(channel_ratio*256),  round(channel_ratio*512),  2),
            conv_dw(round(channel_ratio*512),  round(channel_ratio*512),  1),
            conv_dw(round(channel_ratio*512),  round(channel_ratio*512),  1),
            conv_dw(round(channel_ratio*512),  round(channel_ratio*512),  1),
            conv_dw(round(channel_ratio*512),  round(channel_ratio*512),  1),
            conv_dw(round(channel_ratio*512),  round(channel_ratio*512),  1),
            conv_dw(round(channel_ratio*512),  round(channel_ratio*1024), 2),
            conv_dw(round(channel_ratio*1024), round(channel_ratio*1024), 1),
            # nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(round(channel_ratio*1024), num_classes)
        self._initialize_weights()
        return

    def forward(self, x):
        x = self.model(x)
        # x = x.view(x.size(0), -1)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

    def build_graph(self):

        # Construct the graph
        info = get_module_info(self.model[0][0])
        first_conv_node = Node('model.0.0', 'conv', info)
        graph = Graph(start_node=first_conv_node, name='MobileNetV1')

        info = get_module_info(self.model[0][1])
        first_bn_node = Node('model.0.1', 'bn', info, graph)
        first_conv_node.add_link_to(first_bn_node)

        head_node = first_bn_node
        for block_idx, module in self.model.named_children():
            block_idx = int(block_idx)
            if block_idx == 0: continue

            head_node = parse_dw_conv_to_graph(
                    module, block_idx, head_node, graph)

        info = get_module_info(self.fc)
        fc_node = Node('fc', 'fc', info, graph)
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

def mobilenetv1(num_classes=1000, channel_ratio=1.0):
    model = MobileNetV1(num_classes=num_classes, channel_ratio=channel_ratio)
    return model
