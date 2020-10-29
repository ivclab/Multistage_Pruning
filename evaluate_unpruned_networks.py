import sys
import os
import argparse
import torch
import torch.nn as nn
from train import test
from utils.registry import network_handler, dataset_handler
from utils.graph import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate the unpruned networks')
    parser.add_argument('--chkpt_path', type=str, required=True,
                        help='Path of the stored model.')
    parser.add_argument('--network_name', type=str, required=True,
                        help='Network name for evaluation.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name for evaluation.')
    args = parser.parse_args()
    return args


def main(*args, **kwargs):

    # ----------------------------------------
    # General settings
    # ----------------------------------------
    args = parse_arguments()
    chkpt_path = args.chkpt_path
    network_name = args.network_name
    dataset_name = args.dataset_name

    image_size  = (224, 224)
    batch_size  = 128
    num_classes = 1000

    # ----------------------------------------
    # Model settings
    # ----------------------------------------
    model = network_handler[network_name](num_classes=num_classes)
    chkpt_content = torch.load(chkpt_path)
    model.load_state_dict(chkpt_content['state_dict'])

    # ----------------------------------------
    # Evaluate the unpruned models
    # ----------------------------------------
    device = 'cuda'
    model.to(device)
    model.eval()
    _, test_loader, num_classes = dataset_handler[dataset_name](
            batch_size, batch_size, image_size[0], num_workers=24, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(model, device, test_loader, criterion, -1)
    print('Test Acc@1: {:.4f}, Test Acc@5: {:.4f}'.format(*test_acc))
    return


if __name__ == '__main__':
    main()
