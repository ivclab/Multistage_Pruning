import sys
import imp
import os
import logging
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from shutil import copyfile
import utils.scheduler
from utils.optimizer import MaskedSGD
from utils.pruner import SparsePruner
from utils.tools import *
from utils.registry import network_handler, dataset_handler


def train(model, device, data_loaders, criterion, optimizer, epoch, mode='basline',
          pruner=None):

    assert(mode in ['baseline', 'prune'])
    train_loader, test_loader = data_loaders

    model.train()
    train_loss = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    with tqdm(total=len(train_loader),
              desc='Train Ep. #{}'.format(epoch),
              disable=False,
              ascii=True) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            if mode == 'prune':
                pruner.gradually_prune(device)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            train_loss.update(loss.item(), data.size(0))
            train_top1.update(prec1, data.size(0))
            train_top5.update(prec5, data.size(0))

            tqdm_postfix = {
                'train_acc@1': round(train_top1.avg, 4),
                'train_acc@5': round(train_top5.avg, 4),
                'train_loss':  round(train_loss.avg, 4),
            }

            if mode == 'prune':
                sparsity = pruner.calculate_sparsity()
                tqdm_postfix['sparsity'] = round(sparsity, 4)

            t.set_postfix(tqdm_postfix)
            t.update(1)

    logging.info(('In train() with {} -> Train Ep. #{}: '.format(mode, epoch)
        + ', '.join(['{}: {}'.format(k, v) for k, v in tqdm_postfix.items()])))
    return train_loss.avg, (train_top1.avg, train_top5.avg)


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = AverageMeter()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()

    with tqdm(total=len(test_loader),
              desc='Test Ep. #{}'.format(epoch),
              disable=False,
              ascii=True) as t:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                test_loss.update(loss.item(), data.size(0))
                test_top1.update(prec1, data.size(0))
                test_top5.update(prec5, data.size(0))

                tqdm_postfix = {
                    'test_acc@1': round(test_top1.avg, 4),
                    'test_acc@5': round(test_top5.avg, 4),
                    'test_loss':  round(test_loss.avg, 4),
                }

                t.set_postfix(tqdm_postfix)
                t.update(1)

    logging.info(('In test() -> Test Ep. #{}: '.format(epoch)
        + ', '.join(['{}: {}'.format(k, v) for k, v in tqdm_postfix.items()])))
    return test_loss.avg, (test_top1.avg, test_top5.avg)


def run_baseline(mode_spec, model, device, train_loader, test_loader, chkpt_dir):

    pretrained_path = mode_spec['pretrained'] if ('pretrained' in mode_spec) else ''
    if pretrained_path != '':
        content = load_model(model, pretrained_path)

    optim_type = mode_spec['optim_opt'].pop('type')
    optimizer = getattr(optim, optim_type)(model.parameters(), **mode_spec['optim_opt'])

    epochs = mode_spec['scheduler_opt'].pop('epochs')
    scheduler_type = mode_spec['scheduler_opt'].pop('type')
    if hasattr(utils.scheduler, scheduler_type):
        scheduler = getattr(utils.scheduler, scheduler_type)(optimizer, **mode_spec['scheduler_opt'])
    else:
        scheduler = getattr(lr_scheduler, scheduler_type)(optimizer, **mode_spec['scheduler_opt'])

    model_checkpoint = ModelCheckpoint()
    criterion = nn.CrossEntropyLoss()
    chkpt_path = os.path.join(chkpt_dir, 'baseline.pth')

    # ----------------------------------------
    # Trainval loops
    # ----------------------------------------
    training_records = {'trn_loss': [], 'trn_acc': [], 'tst_loss': [], 'tst_acc': []}
    for epoch in range(1, epochs+1):

        train_loss, train_acc = train(model, device, [train_loader, test_loader], criterion,
                optimizer, epoch, mode='baseline')
        test_loss, test_acc = test(model, device, test_loader, criterion, epoch)

        training_records['trn_loss'].append(train_loss)
        training_records['trn_acc'].append(train_acc)
        training_records['tst_loss'].append(test_loss)
        training_records['tst_acc'].append(test_acc)

        model_checkpoint(test_acc[0], model, epoch, chkpt_path)

        scheduler.step()

    # ----------------------------------------
    # Evaluation
    # ----------------------------------------
    content = load_model(model, chkpt_path)
    print('Evaluate on the test set ... ')
    test_loss, test_acc = test(model, device, test_loader, criterion, content['epoch'])
    print('Test loss: {:.4f}, Test acc (top1): {:.4f}, Test acc (top5): {:.4f}'.format(test_loss, test_acc[0], test_acc[1]))
    save_model(model, chkpt_path, content['epoch'], test_acc=test_acc, training_records=training_records)
    return


def run_prune(mode_spec, model, device, train_loader, test_loader, chkpt_dir, network_name):

    opts = mode_spec['other_opt']
    begin_ratio_list = opts['begin_ratio_list']
    end_ratio_list = opts['end_ratio_list']

    chkpt_path = os.path.join(chkpt_dir, 'ratio_{}.pth'.format(begin_ratio_list[0]))
    if begin_ratio_list[0] == 0.0:
        assert('pretrained' in opts and opts['pretrained'] != '')
        copyfile(opts['pretrained'], chkpt_path)

    prune_freq = opts['prune_freq']
    prune_intv = opts['prune_intv']
    begin_step = 0
    end_step   = begin_step + prune_intv * len(train_loader)
    epochs = mode_spec['scheduler_opt'].pop('epochs')
    scheduler_type = mode_spec['scheduler_opt'].pop('type')
    optim_type = mode_spec['optim_opt'].pop('type')
    assert(optim_type == 'SGD')  # Use MaskedSGD in the following

    content = load_model(model, chkpt_path)
    model_checkpoint = ModelCheckpoint()
    masks, params, mask_mappings, module_dict = init_masks_and_params(model)
    masks = content['masks'] if begin_ratio_list[0] != 0 else masks
    training_records = {}

    # ----------------------------------------
    # Trainval loops
    # ----------------------------------------
    for prune_stage, (begin_ratio, end_ratio) in enumerate(zip(begin_ratio_list, end_ratio_list)):

        pruner = SparsePruner(model, network_name, masks, module_dict, begin_step, end_step,
                              begin_ratio, end_ratio, prune_freq, strategy=opts['strategy'])

        optimizer = MaskedSGD(params, masks=masks, mask_mappings=mask_mappings, **mode_spec['optim_opt'])

        if hasattr(utils.scheduler, scheduler_type):
            scheduler = getattr(utils.scheduler, scheduler_type)(optimizer, **mode_spec['scheduler_opt'])
        else:
            scheduler = getattr(lr_scheduler, scheduler_type)(optimizer, **mode_spec['scheduler_opt'])

        criterion = nn.CrossEntropyLoss()
        chkpt_path = os.path.join(chkpt_dir, 'ratio_{}.pth'.format(end_ratio))
        model_checkpoint.reset()

        print('Pruning from {} to {} ... '.format(begin_ratio, end_ratio))
        training_records[prune_stage] = {'trn_loss': [], 'trn_acc': [], 'tst_loss': [], 'tst_acc': []}
        for epoch in range(1, epochs+1):

            train_loss, train_acc = train(model, device, [train_loader, test_loader], criterion,
                    optimizer, epoch, mode='prune', pruner=pruner)
            test_loss, test_acc = test(model, device, test_loader, criterion, epoch)

            training_records[prune_stage]['trn_loss'].append(train_loss)
            training_records[prune_stage]['trn_acc'].append(train_acc)
            training_records[prune_stage]['tst_loss'].append(test_loss)
            training_records[prune_stage]['tst_acc'].append(test_acc)

            if epoch > prune_intv:
                model_checkpoint(test_acc[0], model, epoch, chkpt_path, masks=masks)

            scheduler.step()

        content = load_model(model, chkpt_path)
        print('Evaluate pruning ratio {} on the test set ...'.format(end_ratio))
        test_loss, test_acc = test(model, device, test_loader, criterion, content['epoch'])
        print('Test loss: {:.4f}, Test acc (top1): {:.4f}, Test acc (top5): {:.4f}'.format(test_loss, test_acc[0], test_acc[1]))
        save_model(model, chkpt_path, content['epoch'], masks=content['masks'], test_acc=test_acc,
                   training_records=training_records)
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Structrue pruning pytorch implementation')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='To disable CUDA training.')
    parser.add_argument('--seed', type=int, default=0xCAFFE,
                        help='Manual setting of RNG seeds.')
    parser.add_argument('--config_name', type=str, required=True,
                        help='Configuration name.')
    parser.add_argument('--target_mode', type=str, required=True,
                        help='Target mode specified in configuration.')
    parser.add_argument('--postfix', type=str, default='',
                        help='Postfix of the config name.')
    args = parser.parse_args()
    return args


def main(*args, **kwargs):

    # ----------------------------------------
    # General settings
    # ----------------------------------------
    args = parse_arguments()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    chkpt_dir = 'CHECKPOINTS/{}{}/{}'.format(args.config_name, args.postfix, args.target_mode)
    os.makedirs(chkpt_dir, exist_ok=True)
    set_logger(os.path.join(chkpt_dir, 'run.log'))

    # Convert the pretrained mobilenetv2 from torchvision
    if args.config_name == 'MobileNetV2_ImageNet' and args.target_mode == 'baseline':

        from torchvision.models import mobilenet
        train_loader, test_loader, num_classes = dataset_handler['ImageNet'](
                128, 128, 224, num_workers=24, pin_memory=True)

        orig_model = mobilenet.mobilenet_v2(pretrained=True)
        model = network_handler['MobileNetV2'](num_classes=1000)
        model = model.to(device)
        model.load_state_dict(orig_model.state_dict())
        criterion = nn.CrossEntropyLoss()

        test_loss, test_acc = test(model, device, test_loader, criterion, -1)
        content = {}
        content['state_dict'] = model.state_dict()
        content['test_acc'] = test_acc
        content['training_records'] = {'trn_loss': [], 'trn_acc': [], 'tst_loss': [], 'tst_acc': []}
        content['epoch'] = -1

        chkpt_dir = 'CHECKPOINTS/MobileNetV2_ImageNet/baseline'
        os.makedirs(chkpt_dir, exist_ok=True)

        torch.save(content, os.path.join(chkpt_dir, 'baseline.pth'))

        return  # Don't need to train after model conversion

    config = imp.load_source('', 'configs/'+args.config_name+'.py').config
    mode_spec = config['modes'][args.target_mode]
    assert(mode_spec['type'] in ['baseline', 'prune'])

    # ----------------------------------------
    # Build the data loaders
    # ----------------------------------------
    dataset_name = config['dataset']
    batch_size = mode_spec['other_opt']['batch_size']
    train_loader, test_loader, num_classes = dataset_handler[dataset_name](
            batch_size, batch_size, mode_spec['other_opt']['image_size'],
            num_workers=mode_spec['other_opt']['num_workers'], pin_memory=True)

    # ----------------------------------------
    # Build the network
    # ----------------------------------------
    network_name = config['network']
    model = network_handler[network_name](num_classes=num_classes)
    if use_cuda:
        model = nn.DataParallel(model)
    model = model.to(device)
    print(model)

    # ----------------------------------------
    # Runing trainval loops
    # ----------------------------------------
    if mode_spec['type'] == 'baseline':
        run_baseline(mode_spec, model, device, train_loader, test_loader, chkpt_dir)
    else:
        run_prune(mode_spec, model, device, train_loader, test_loader, chkpt_dir, network_name)
    return


if __name__ == "__main__":
    main()
