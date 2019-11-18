import random
from torchvision import datasets, transforms
from torch_tools.data import LabeledDatasetImagesExtractor, UnannotatedDataset, TransformedDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset


DATASETS = {
    'ImageNet': 'path to imagenet gans dir',
    'cifar10': 'path to cifar gans dir',
    'lsun_bedroom': 'path to lsun gans dir',
    'MNIST': 'path to mnist gans dir',
}


def transform_for_dataset(name, size, original=True):
    shift_to_zero = lambda x: 2 * x - 1
    id_transform = lambda x: x

    if 'ImageNet' in name:
        def central_crop(x):
            dims = x.size
            crop = transforms.CenterCrop(min(dims[0], dims[1]))
            return crop(x)

        transform = transforms.Compose([
            central_crop,
            transforms.Resize(size) if size is not None else transforms.Resize(256),
            transforms.ToTensor(),
            shift_to_zero])
    elif 'cifar10' in name:
        transform = transforms.Compose([
            transforms.Resize(size) if size is not None else transforms.Resize(32),
            transforms.ToTensor(),
            shift_to_zero])
    elif 'lsun_bedroom' in name:
        transform = transforms.Compose([
            transforms.Resize([size, size]) if size is not None else transforms.Resize([256, 256]),
            transforms.ToTensor(),
            shift_to_zero])
    elif 'MNIST' in name:
        transform = transforms.Compose([
            transforms.Resize(size) if size is not None else id_transform,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            lambda x: x.repeat([3, 1, 1])
        ])
    else:
        raise KeyError('unsupported dataset')

    return transform


def to_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=1, drop_last=True)


def make_dataset_loader(name, batch_size, size=None, data_dir=None, samples_to_take=None,
                        shuffle=True, drop_last=True):
    transform = transform_for_dataset(name, size, True)
    if data_dir is None:
        data_dir = DATASETS[name]

    if 'ImageNet' in name:
        ds = LabeledDatasetImagesExtractor(
            datasets.ImageFolder(root=data_dir, transform=transform))
    elif 'cifar10' in name:
        ds = LabeledDatasetImagesExtractor(
            datasets.CIFAR10(root=data_dir, download=True, transform=transform))
    elif 'lsun_bedroom' in name:
        ds = LabeledDatasetImagesExtractor(
            datasets.LSUN(data_dir, classes=['bedroom_train'], transform=transform))
    elif 'MNIST' in name:
        ds = LabeledDatasetImagesExtractor(datasets.MNIST(data_dir, transform=transform))
    else:
        ds = UnannotatedDataset(data_dir, sorted=False, transform=transform)

    if samples_to_take is not None:
        if len(ds) < samples_to_take:
            print('dataset is too small, repeating it')
            repeats = samples_to_take // len(ds) + 1
            ds = ConcatDataset([ds] * repeats)

        all_indices = list(range(len(ds)))
        random.shuffle(all_indices)
        ds = Subset(ds, all_indices[:samples_to_take])

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=drop_last)
