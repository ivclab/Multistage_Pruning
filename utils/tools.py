import sys
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn


def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    sh.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return


def save_model(model, checkpoint_path, epoch, test_acc=None, masks=None,
               training_records=None):
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()
    model_content = {'state_dict': model_state_dict}
    if test_acc is not None:
        model_content['test_acc'] = test_acc
    if training_records is not None:
        model_content['training_records'] = training_records
    if masks is not None:
        model_content['masks'] = masks
    model_content['epoch'] = epoch
    torch.save(model_content, checkpoint_path)
    return


def load_model(model, checkpoint_path):
    model_content = torch.load(checkpoint_path)
    state_dict = model_content['state_dict']
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()
    loaded_keys = []
    skip_keys = []
    missing_keys = []
    for name, params in state_dict.items():
        if name in model_state_dict and model_state_dict[name].size() == params.size():
            model_state_dict[name].data.copy_(params)
            loaded_keys.append(name)
        else:
            skip_keys.append(name)
    missing_keys = list(set(model_state_dict.keys()) - set(loaded_keys))
    print('In the model-> missing keys:')
    print('\t' + ', '.join(np.sort(missing_keys)))
    print('In the checkpoint-> skip keys: ')
    print('\t' + ', '.join(np.sort(skip_keys)))
    other_content = {}
    for k, v in model_content.items():
        if k == 'state_dict': continue
        other_content[k] = v
    return other_content


def init_masks_and_params(model):
    masks, params = {}, []
    mask_mappings = {}

    # Collections of leaf modules
    module_dict = { name.replace('module.', ''): module
                    for name, module in model.named_modules()
                    if len(module._modules) == 0 }

    for index, (name, param) in enumerate(model.named_parameters()):
        params.append(param)
        split_idx = name.rfind('.')
        module_name = name[:split_idx].replace('module.', '')
        weight_name = name[split_idx+1:]
        module = module_dict[module_name]

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if weight_name == 'weight':
                masks[module_name] = torch.ones(param.size())
                mask_mappings[index] = module_name

        elif isinstance(module, nn.BatchNorm2d):
            masks[module_name] = torch.ones(param.size())
            mask_mappings[index] = module_name

        else:
            raise NotImplementedError('Unexpected errors occur')

    parmas = {'params': params}
    return masks, params, mask_mappings, module_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class ModelCheckpoint(object):
    def __init__(self):
        self.best_score = None
        return

    def __call__(self, score, model, epoch, checkpoint_path, masks=None):
        chkpt_split = os.path.splitext(checkpoint_path)
        save_model(model, chkpt_split[0]+'-checkpoint'+chkpt_split[1], epoch,
                   masks=masks)
        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            save_model(model, checkpoint_path, epoch, masks=masks)
            print('Improvement!')
        return

    def reset(self):
        self.best_score = None
        return
