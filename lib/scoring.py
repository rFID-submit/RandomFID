import os
import sys
from random import shuffle
import argparse
import json
import torch
from torch.utils.data import Subset
import numpy as np

from torch_tools.data import UnannotatedDataset

from fid_models.models_factory import make_random_model
from fid_models.inception import InceptionV3
from fid_models.lenet import make_random_lenet
from fid_models.model_activations import model_activations
from fid_score import calculate_fid_for_dataloaders
from data import make_dataset_loader, to_dataloader, transform_for_dataset, DATASETS


class Params():
    def __init__(self, **kwargs):
        self.runs = 5
        self.gans = None
        self.dataset = None
        self.size = None
        self.batch = 256
        self.dataset_dir = None
        self.samples_to_take = 1e+4
        self.random_init = None
        self.model = None
        self.steps_for_fit = 50
        self.no_bias = False
        self.ref_model = None

        self.__dict__.update(kwargs)


def make_ref_fid_model(name, resized=False):
    if name == 'LeNet':
        return make_random_lenet(None, pretrained=True)
    elif name == 'inception_v3_mnist':
        return model_activations('inception_v3_mnist',
            pretrained='./pretrained/inception_v3_mnist.pt')
    elif name == 'inception_v3_cifar10':
        return model_activations('inception_v3_cifar10',
            pretrained='./pretrained/inception_v3_cifar10.pt')
    elif name == 'inception_v3':
        return InceptionV3(resize_input=resized)


def compare_fids(gans_samples_dir, dataset, size, model, ref_model,
        samples_to_take, batch, random_init, runs, ref_data_dir, out_dir, steps_for_fit, no_bias,
        linear=False):
    assert samples_to_take > batch, 'too few samples to take'
    debug_dir = os.path.join(out_dir, 'statistics')

    if ref_data_dir is None:
        ref_data_dir = DATASETS[dataset]

    original_dataloader = make_dataset_loader(
        dataset, batch, size=size, data_dir=ref_data_dir, samples_to_take=samples_to_take)

    gans_samples_dirs = \
        [os.path.join(gans_samples_dir, subdir) for subdir in os.listdir(gans_samples_dir)]
    gans_samples_dirs = [s for s in gans_samples_dirs if os.path.isdir(s)]

    transform = transform_for_dataset(dataset, size, False)
    gan_samples_dataloaders = []
    for data_dir in gans_samples_dirs:
        ds_full = UnannotatedDataset(data_dir, transform=transform)
        indices = np.arange(len(ds_full))
        shuffle(indices)
        indices = indices[:samples_to_take]

        gan_samples_dataloaders.append(to_dataloader(
                Subset(ds_full, indices), batch))

    all_dls = gan_samples_dataloaders + [original_dataloader]
    out_dict = {'dirs': gans_samples_dirs}

    # FIDs calculation
    if ref_model is not None:
        print('computing original FID')
        original_inception = make_ref_fid_model(ref_model)
        original_inception.cuda()
        fids_ref = calculate_fid_for_dataloaders(
            original_dataloader, all_dls, model=original_inception, debug_dir=debug_dir)
        out_dict['fids_original'] = fids_ref

    if model is not None:
        fids_random = []
        for i in range(runs):
            print('random {} FID iteration: {}'.format(model, i))
            try:
                random_model = make_random_model(model, original_dataloader, random_init,
                                                 steps_for_fit=steps_for_fit, no_bias=no_bias,
                                                 linear=linear)
                fids_random.append(calculate_fid_for_dataloaders(
                        original_dataloader, all_dls, model=random_model, debug_dir=debug_dir))
            except NameError as e:
                print('FAILED: {}'.format(e))
        out_dict['fids_random'] = fids_random

    with open(os.path.join(out_dir, 'fids.json'), 'w') as out:
        json.dump(out_dict, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrained inception vs random FID')
    parser.add_argument('--out', type=str, help='out fids dir')
    parser.add_argument('--gans', type=str, help='root directory for gans samples')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of independent random Inceptions models runs')
    parser.add_argument('--dataset', type=str, choices=DATASETS, help='refernce dataset')
    parser.add_argument('--size', type=int, default=None, help='target images size')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--dataset_dir', type=str, default=None, help='refernce dataset dir')
    parser.add_argument('--samples_to_take', type=int, default=1e+4,
                        help='samples for FID calculation')
    parser.add_argument('--device', type=int, default=0, help='cuda device to use')
    parser.add_argument('--random_init', type=str, default=None, help='weights initializer')
    parser.add_argument('--model', type=str, default=None, help='feature extractor')
    parser.add_argument('--steps_for_fit', type=int, default=50, help='steps to fit random model')
    parser.add_argument('--no_bias', action='store_true', default=False,
                        help='remove bias from random model')
    parser.add_argument('--linear', action='store_true', default=False,
                        help='make random model linear')
    parser.add_argument('--ref_model', type=str, default=None, help='pretrained feature extractor')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    print('running {} <- {} -> {} @ device {}'.format(
        args.gans, args.model, args.dataset, args.device))

    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    compare_fids(gans_samples_dir=args.gans,
                 dataset=args.dataset,
                 size=args.size,
                 model=args.model,
                 ref_model=args.ref_model,
                 samples_to_take=args.samples_to_take,
                 batch=args.batch,
                 random_init=args.random_init,
                 runs=args.runs,
                 ref_data_dir=args.dataset_dir,
                 out_dir=args.out,
                 steps_for_fit=args.steps_for_fit,
                 no_bias=args.no_bias,
                 linear=args.linear)

    print('done ({})'.format(args.gans))