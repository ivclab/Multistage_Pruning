import math
from torch.optim import lr_scheduler


class EpochBasedExponentialLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay=0.96, last_epoch=-1):
        self.decay = decay
        super(EpochBasedExponentialLR, self).__init__(optimizer, last_epoch)
        return

    def get_lr(self):
        return [base_lr * (self.decay ** self.last_epoch)
                for base_lr in self.base_lrs]


class EpochBasedCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, last_epoch=-1):
        self.max_epochs = max_epochs
        super(EpochBasedCosineLR, self).__init__(optimizer, last_epoch)
        return

    def get_lr(self):
        return [0.5 * base_lr * (1 + math.cos(math.pi * self.last_epoch / self.max_epochs))
                for base_lr in self.base_lrs]
