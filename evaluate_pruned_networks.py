import sys
import os
import argparse
import torch
import torch.nn as nn
from thop import profile
from train import test
from utils.registry import network_handler, dataset_handler
from utils.graph import *


def trace_add_node(add_node):
    assert(len(add_node.prev_nodes) > 0)
    for node in add_node.prev_nodes:
        if isinstance(node, AddNode):
            non_add_node = trace_add_node(node)
        else:
            non_add_node = node
            break
    return non_add_node


def compute_out_idx_from_mask(mask):
    if mask.dim() == 4:
        output_idx = torch.sum(mask, (1, 2, 3)).nonzero().squeeze(-1)
    elif mask.dim() == 2:
        output_idx = torch.sum(mask, 1).nonzero().squeeze(-1)
    elif mask.dim() == 1:
        output_idx = mask.nonzero().squeeze(-1)
    else:
        raise NotImplementedError(
            'Unexpected mask dimensions: `{}`'.format(mask.dim()))
    return output_idx


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evalaute the pruned networks')
    parser.add_argument('--network_name', type=str, required=True,
                        help='Network name for evaluation.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name for evaluation.')
    parser.add_argument('--target_mode', type=str, required=True,
                        help='Target mode of pruning.')
    parser.add_argument('--prune_ratio', type=float, required=True,
                        help='Prune ratio of the pruned network.')
    args = parser.parse_args()
    return args


def main(*args, **kwargs):

    # ----------------------------------------
    # General settings
    # ----------------------------------------
    args = parse_arguments()
    network_name = args.network_name
    dataset_name = args.dataset_name
    target_mode = args.target_mode
    prune_ratio = args.prune_ratio
    assert(prune_ratio > 0.0), 'Only evaluate networks with prune_ratios > 0.0'

    image_size  = (224, 224)
    batch_size  = 128
    num_classes = 1000
    chkpt_path  = os.path.join('CHECKPOINTS/{}_{}/{}/ratio_{}.pth'.format(
        network_name, dataset_name, target_mode, prune_ratio))

    # ----------------------------------------
    # Model settings
    # ----------------------------------------
    origin_model = network_handler[network_name](num_classes=num_classes)
    pruned_model = network_handler[network_name](num_classes=num_classes, channel_ratio=1.0-prune_ratio)
    chkpt_content = torch.load(chkpt_path)
    origin_model.load_state_dict(chkpt_content['state_dict'])
    masks = chkpt_content['masks']
    graph = origin_model.build_graph()
    origin_state_dict = origin_model.state_dict()

    for name, params in pruned_model.named_parameters():
        device = params.device
        module_name = name[:name.rfind('.')]
        curr_node = graph.node_dict[module_name]

        assert(len(curr_node.prev_nodes) in [0, 1])
        prev_node = None if (len(curr_node.prev_nodes) == 0) else curr_node.prev_nodes[0]

        if isinstance(prev_node, AddNode):
            prev_node = trace_add_node(prev_node)

        if curr_node.module_type == 'conv':
            assert(params.dim() in [1, 4])
            out_chs = compute_out_idx_from_mask(masks[curr_node.module_name]).to(device)
            if params.dim() == 4:  # weight
                if prev_node is not None:
                    inp_chs = compute_out_idx_from_mask(masks[prev_node.module_name]).to(device)

                pruned_params = origin_state_dict[name].index_select(0, out_chs)
                if (prev_node is not None) and (curr_node.info['groups'] == 1):  # Non-First Layer and Non-Depthwise Conv
                    pruned_params = pruned_params.index_select(1, inp_chs)

            else:                  # bias
                pruned_params = origin_state_dict[name].index_select(0, out_chs)

        elif curr_node.module_type == 'bn':
            assert(params.dim() == 1)
            out_chs = compute_out_idx_from_mask(masks[curr_node.module_name]).to(device)
            pruned_params = origin_state_dict[name].index_select(0, out_chs)

        elif curr_node.module_type == 'fc':
            assert(params.dim() in [1, 2])
            out_chs = compute_out_idx_from_mask(masks[curr_node.module_name]).to(device)
            if params.dim() == 2:  # weight
                if prev_node is not None:
                    inp_chs = compute_out_idx_from_mask(masks[prev_node.module_name]).to(device)

                pruned_params = origin_state_dict[name].index_select(0, out_chs)
                if prev_node is not None:
                    pruned_params = pruned_params.index_select(1, inp_chs)

            else:                  # bias
                pruned_params = origin_state_dict[name].index_select(0, out_chs)

        else:
            raise NotImplementedError('Unexpected module_type `{}`'.format(
                curr_node.module_type))

        params.data.copy_(pruned_params)

    for name, buffers in pruned_model.named_buffers():
        device = params.device
        module_name = name[:name.rfind('.')]
        curr_node = graph.node_dict[module_name]

        assert(curr_node.module_type == 'bn')
        if 'num_batches_tracked' in name:
            buffers.data.copy_(origin_state_dict[name])
        else:
            out_chs = compute_out_idx_from_mask(masks[curr_node.module_name]).to(device)
            pruned_buffers = origin_state_dict[name].index_select(0, out_chs)
            buffers.data.copy_(pruned_buffers)

    # ----------------------------------------
    # Evaluate the origin and pruned models
    # ----------------------------------------
    device = 'cuda'
    origin_model.to(device)
    pruned_model.to(device)
    origin_model.eval()
    pruned_model.eval()
    _, test_loader, num_classes = dataset_handler[dataset_name](
            batch_size, batch_size, image_size[0], num_workers=24, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(pruned_model, device, test_loader, criterion, -1)
    print('Test Acc@1: {:.4f}, Test Acc@5: {:.4f}'.format(*test_acc))

    images = torch.randn(*((1, 3)+image_size)).to(device)
    flops1, params1 = profile(origin_model, inputs=(images,), verbose=False)
    flops2, params2 = profile(pruned_model, inputs=(images,), verbose=False)

    print('Full Network FLOPs: {:.4f} M, Params: {:.4f} M'.format(flops1/(1000**2), params1/(1000**2)))
    print('Pruned Network FLOPs: {:.4f} M, Params: {:.4f} M'.format(flops2/(1000**2), params2/(1000**2)))
    return


if __name__ == '__main__':
    main()
