"""Image transforms shared by training and evaluation notebooks."""

from torchvision import transforms


def get_transforms(train: bool, mean: tuple[float, ...], std: tuple[float, ...]):
    """Build CIFAR-sized training augmentations or deterministic evaluation transforms."""
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize])
