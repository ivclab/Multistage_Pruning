from .datasets import *
from .networks import *


dataset_handler = {
    'ImageNet': make_imagenet_dataloader,
}


network_handler = {
    'MobileNetV1': mobilenetv1,
    'MobileNetV2': mobilenetv2,
}
