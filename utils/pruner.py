import math
import sys
import numpy as np
import torch
import torch.nn as nn
from utils.graph import *


def zero_out_masks_and_weights(constraints, masks, module_dict, cutoff_idx_map):

    def prune_next_layer_kernels(node, cutoff_idx):
        for next_node in node.next_nodes:
            if ((next_node.module_type == 'conv' and next_node.info['groups'] == 1)
                or next_node.module_type == 'fc'):
                masks[next_node.module_name].index_fill_(1, cutoff_idx.cpu(), 0)
                module = module_dict[next_node.module_name]
                mask = masks[next_node.module_name]
                module.weight.data[mask.eq(0)] = 0.0
            elif next_node.module_type == 'add':
                prune_next_layer_kernels(next_node, cutoff_idx)
            elif (next_node.module_type == 'bn'
                  or (next_node.module_type == 'conv' and next_node.info['groups'] != 1)):
                pass  # No input dimensions for bn or depth-wise conv to prune
            else:
                print(next_node.module_type)
                raise NotImplementedError('Unexpected errors occur')
        return

    # Zero out masks, current layers' filters and next layers' kernels
    for constraint_name, cluster in constraints.items():
        if len(cluster) == 1 and cluster[0].module_type == 'fc': continue  # Don't prune fc
        cutoff_idx = cutoff_idx_map[constraint_name]

        for node in cluster:

            # Current layers' filters
            masks[node.module_name][cutoff_idx] = 0
            module = module_dict[node.module_name]
            mask = masks[node.module_name]
            if node.module_type == 'conv':
                module.weight.data[mask.eq(0)] = 0.0
                if module.bias:
                    module.bias.data[torch.sum(mask, (1, 2, 3)).eq(0)] = 0.0
            elif node.module_type == 'bn':
                module.weight.data[mask.eq(0)] = 0.0
                module.bias.data[mask.eq(0)] = 0.0
                module.running_mean.data[mask.eq(0)] = 0.0
                module.running_var.data[mask.eq(0)] = 0.0
            else:
                raise NotImplementedError(
                    'Unexpected errors occur')

            # Next layers' kernels
            prune_next_layer_kernels(node, cutoff_idx)
    return


def magnitude_sum_min_filter_pruning(constraints, masks, module_dict, device, prune_ratio, use_bn=False):

    # Compute abs sum for every filters
    cutoff_idx_map = {}
    for constraint_name, cluster in constraints.items():
        if len(cluster) == 1 and cluster[0].module_type == 'fc': continue  # Don't prune fc

        # Compute abs weight scores
        num_filters = module_dict[cluster[0].module_name].weight.size(0)
        abs_sum = torch.zeros((num_filters,)).to(device)
        for node in cluster:
            if node.module_type == 'conv':
                abs_weight = torch.sum(module_dict[node.module_name].weight.data.abs(), (1, 2, 3))
                next_bn = len(node.next_nodes) == 1 and node.next_nodes[0].module_type == 'bn'
                if use_bn and next_bn:
                    bn_module_name = node.next_nodes[0].module_name
                    bn_scale = module_dict[bn_module_name].weight.data.abs()
                    abs_weight *= bn_scale
                abs_sum += abs_weight

        # Prune the filters with smallest scores
        cutoff_rank = round(prune_ratio*abs_sum.numel())
        if cutoff_rank != 0:
            cutoff_value = abs_sum.cpu().kthvalue(cutoff_rank)[0].cuda()
            cutoff_idx = abs_sum.le(cutoff_value).nonzero().squeeze(-1)
        else:
            cutoff_idx = abs_sum.new_tensor([]).type(torch.long)
        cutoff_idx_map[constraint_name] = cutoff_idx

    zero_out_masks_and_weights(constraints, masks, module_dict, cutoff_idx_map)
    return


class SparsePruner(object):
    def __init__(self, model, network_name, masks, module_dict, begin_step, end_step, begin_ratio, end_ratio, prune_freq,
                 strategy='magnitude-sum-min', use_bn=False):

        self.model = model
        self.network_name = network_name
        self.masks = masks
        self.module_dict = module_dict
        self.begin_step = begin_step
        self.end_step   = end_step
        self.begin_ratio = begin_ratio
        self.end_ratio   = end_ratio
        self.last_prune_step = begin_step
        self.prune_freq = prune_freq
        self.ratio_func_exp = 3
        self.curr_prune_step = 0
        self.strategy = strategy
        self.use_bn = use_bn

        self.graph = model.module.build_graph()
        self.constraints = investigate_constraints(self.graph, show=True)
        assert self.strategy in ['magnitude-sum-min']
        return

    def compute_prune_ratio(self, prune_step):
        p = min(1.0, max(0.0, (prune_step - self.begin_step)/(self.end_step - self.begin_step)))
        ratio = self.end_ratio + (self.begin_ratio - self.end_ratio)*pow(1-p, self.ratio_func_exp)
        return ratio

    def time_to_prune(self):
        is_pruning_range = (self.curr_prune_step >= self.begin_step) and (self.curr_prune_step <= self.end_step)
        is_pruning_step = (self.last_prune_step + self.prune_freq) <= self.curr_prune_step
        return is_pruning_range and is_pruning_step

    def gradually_prune(self, device):
        if self.time_to_prune():
            self.last_prune_step = self.curr_prune_step
            curr_prune_ratio = self.compute_prune_ratio(self.curr_prune_step)

            # Pruning convolution filters (c_out)
            if self.network_name in ['MobileNetV1', 'MobileNetV2']:
                self.gradually_prune_output_channel(device, curr_prune_ratio)
            else:
                raise NotImplementedError(
                    'Unsupport arch {} for output channel pruning'.format(self.network_name))
        else:
            curr_prune_ratio = self.compute_prune_ratio(self.last_prune_step)

        self.curr_prune_step += 1
        return curr_prune_ratio

    def preview_pruning_ratios(self):
        prune_ratios = []
        last_s = self.begin_step
        for s in range(self.begin_step, self.end_step+1):
            is_pruning_range = (s >= self.begin_step) and (s <= self.end_step)
            is_pruning_step = (last_s + self.prune_freq) <= s
            if is_pruning_range and is_pruning_step:
                last_s = s
                prune_ratios.append(self.compute_prune_ratio(s))
            else:
                if len(prune_ratios) == 0:
                    prune_ratios.append(self.begin_ratio)
                else:
                    prune_ratios.append(prune_ratios[-1])
        prune_ratios = np.unique(prune_ratios).tolist()
        return prune_ratios

    def calculate_sparsity(self):
        total_elem = 0
        zero_elem = 0
        for name, mask in self.masks.items():
            total_elem += mask.numel()
            zero_elem += torch.sum(mask.eq(0)).item()
        return zero_elem / total_elem

    def gradually_prune_output_channel(self, device, ratio):
        if self.strategy == 'magnitude-sum-min':
            magnitude_sum_min_filter_pruning(
                    self.constraints, self.masks, self.module_dict, device, ratio,
                    use_bn=self.use_bn)
        else:
            raise NotImplementedError('Unsupport strategy: {}'.format(self.strategy))
        return
