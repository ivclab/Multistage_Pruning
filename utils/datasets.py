import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def make_imagenet_dataloader(train_batch_size, test_batch_size, image_size, **kwargs):
    assert(image_size == -1 or image_size == 224), 'Currently we only use default (224x224) for imagenet'
    num_classes = 1000
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    input_size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder('DATA/ImageNet/train', train_transform)
    test_dataset = datasets.ImageFolder('DATA/ImageNet/test', test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_classes
