from copy import deepcopy
config = {'network': 'MobileNetV2', 'dataset': 'ImageNet', 'modes': {}}


# -----------------------------------------------
# 8-stage channel pruning magnitude-sum-min
# -----------------------------------------------
prune_with_8stage = {'type': 'prune'}

optim_opt = {}
optim_opt['type'] = 'SGD'
optim_opt['lr'] = 0.01
optim_opt['momentum'] = 0.9
optim_opt['weight_decay'] = 4e-5
optim_opt['nesterov'] = False
prune_with_8stage['optim_opt'] = optim_opt

scheduler_opt = {}
scheduler_opt['epochs'] = 30
scheduler_opt['type'] = 'EpochBasedExponentialLR'
scheduler_opt['decay'] = 0.96
prune_with_8stage['scheduler_opt'] = scheduler_opt

other_opt = {}
other_opt['image_size'] = 224
other_opt['batch_size'] = 256
other_opt['num_workers'] = 24
other_opt['prune_freq'] = 2000
other_opt['prune_intv'] = 4
other_opt['strategy'] = 'magnitude-sum-min'
other_opt['begin_ratio_list'] = [0.000,0.125,0.250,0.375,0.500,0.625,0.750]
other_opt['end_ratio_list']   = [0.125,0.250,0.375,0.500,0.625,0.750,0.875]
other_opt['pretrained'] = 'CHECKPOINTS/MobileNetV2_ImageNet/baseline/baseline.pth'
prune_with_8stage['other_opt'] = other_opt

config['modes']['8stage_prune-magnitude'] = prune_with_8stage


# -----------------------------------------------
# 16-stage channel pruning magnitude-sum-min
# -----------------------------------------------
prune_with_16stage = {'type': 'prune'}

optim_opt = {}
optim_opt['type'] = 'SGD'
optim_opt['lr'] = 0.01
optim_opt['momentum'] = 0.9
optim_opt['weight_decay'] = 4e-5
optim_opt['nesterov'] = False
prune_with_16stage['optim_opt'] = optim_opt

scheduler_opt = {}
scheduler_opt['epochs'] = 15
scheduler_opt['type'] = 'EpochBasedExponentialLR'
scheduler_opt['decay'] = 0.96
prune_with_16stage['scheduler_opt'] = scheduler_opt

other_opt = {}
other_opt['image_size'] = 224
other_opt['batch_size'] = 256
other_opt['num_workers'] = 24
other_opt['prune_freq'] = 2000
other_opt['prune_intv'] = 4
other_opt['strategy'] = 'magnitude-sum-min'
other_opt['begin_ratio_list'] = [0.0000,0.0625,0.1250,0.1875,0.2500,0.3125,0.3750,0.4375,0.5000,0.5625,0.6250,0.6875,0.7500,0.8125,0.8750]
other_opt['end_ratio_list']   = [0.0625,0.1250,0.1875,0.2500,0.3125,0.3750,0.4375,0.5000,0.5625,0.6250,0.6875,0.7500,0.8125,0.8750,0.9375]
other_opt['pretrained'] = 'CHECKPOINTS/MobileNetV2_ImageNet/baseline/baseline.pth'
prune_with_16stage['other_opt'] = other_opt

config['modes']['16stage_prune-magnitude'] = prune_with_16stage
