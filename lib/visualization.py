import os
import re
import operator
import json
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import itertools
from functools import cmp_to_key

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch_tools.data import UnannotatedDataset
from torch_tools.visualization import to_image
from data import make_dataset_loader


def sort_by(permutation, target):
    return [target[i] for i in permutation]


def value_in_string(s):
    num = re.sub('\D', '', s)
    return 0 if len(num) == 0 else int(num)


def sort_by_values(v):
    indices = list(range(len(v)))
    indices.sort(key=lambda i: v[i])
    return indices


def sort_by_votes(votes):
    votes = np.array(votes)
    indices = np.arange(0, votes.shape[1], 1)
    def cmp(i, j):
        votes_i, votes_j = 0, 0
        for vote in votes:
            if vote[i] > vote[j]:
                votes_i += 1
            else:
                votes_j += 1
        return np.sign(votes_i - votes_j)
    indices = sorted(indices, key=cmp_to_key(cmp))
    return indices


def prob_to_disorder(v1, v2):
    size = len(v1)
    fails = 0
    for i, j in itertools.product(range(size), range(size)):
        if i != j:
            fails += np.sign(v1[i] - v1[j]) != np.sign(v2[i] - v2[j])

    return float(fails) / (size * (size - 1))


def mean_shift(v1, v2):
    ref_permutation = np.array(sort_by_values(v1))
    permutation = np.array(sort_by_values(v2))

    avg = 0.0
    size = len(permutation)
    for i in range(size):
        pos = np.where(permutation == i)[0][0]
        pos_ref = np.where(ref_permutation == i)[0][0]
        avg += abs(pos - pos_ref)
    
    return avg / size


def compare_sorts(v1, v2):
    size = len(v1)
    sorted1, sorted2 = sort_by_values(v1), sort_by_values(v2)
    matches = np.sum([sorted1[i] == sorted2[i] for i in range(size)])
    print(float(matches) / size)


def load_fids_from_dir(root_dir, key):
    with open(os.path.join(root_dir, 'fids.json'), 'r') as source:
        info = json.load(source)
    if not key in info:
        return None, None
    fids = np.array(info[key])
    pathes = info['dirs']

    return fids, pathes


def inspect_random_fid_evaluation(rand_models_dir, ref_model_dir,
        axes=None, sort_generators=False, ref_penalty=None):
    fids_random, pathes_random = load_fids_from_dir(rand_models_dir, 'fids_random')
    fids_original, pathes_original = load_fids_from_dir(ref_model_dir, 'fids_original')

    if fids_random is None or fids_original is None or pathes_random != pathes_original:
        print('skip {}'.format(os.path.basename(rand_models_dir)))
        return

    pathes = pathes_random + ['ref']
    names = [os.path.basename(p) for p in pathes]

    if axes is None:
        _, axes = plt.subplots(1, 3)

    plt.sca(axes[0])
    axes[0].grid(True)
    plt.tick_params(labelbottom=False)
    plt.xticks(range(len(names)), names)

    for i, fids in enumerate(fids_random):
        plt.plot(fids / fids_random.max(), color=plt.cm.autumn(255 * i // len(fids_random)))

    plt.plot(fids_original / fids_original.max(), color='Green', label='with resize')
    plt.legend()

    plt.sca(axes[1])
    axes[1].grid(True)
    plt.xticks(range(len(names)), names)

    fids_random_summarized = fids_random.mean(axis=0)
    plt.plot(fids_random_summarized[:-1] / fids_random_summarized.max(), color='Red')
    plt.plot(fids_original[:-1] / fids_original.max(), color='Green')

    log_std = np.log(fids_random[:, :-1]).std(axis=0)
    plt.plot(log_std, color='Orange')

    plt.sca(axes[2])
    plt.scatter(x=fids_original[:-1], y=fids_random_summarized[:-1], s=4)

    summary = \
        'swap prob: {:.2}\n\
        {} mean shift of {}\n\
        {} original\n\
        {} rand _avg\n\
        {} rand by votes\n\
        std: {:.2}'.format(
            prob_to_disorder(fids_random_summarized[:-1], fids_original[:-1]),
            mean_shift(fids_original, fids_random_summarized), len(fids_original),
            sort_by(sort_by_values(fids_original), names),
            sort_by(sort_by_values(fids_random_summarized), names),
            sort_by(sort_by_votes(fids_random), names),
            log_std.mean()
    )
    if ref_penalty is not None:
        summary += '\n\nref swap prob rand: {} | vanila: {}'.format(
            prob_to_disorder(fids_random_summarized[:-1], ref_penalty),
            prob_to_disorder(fids_original[:-1], ref_penalty)
        )
    axes[2].legend([summary], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def inspect_fids(root_dir, ref_dir=None, scale=1.0, ref_penalty=None):
    models_names = [n for n in os.listdir(root_dir) if \
        os.path.isdir(os.path.join(root_dir, n)) and n != 'ref']
    models_names.sort(key=value_in_string)
    models_dirs = [os.path.join(root_dir, n) for n in models_names\
        if os.path.isfile(os.path.join(root_dir, n, 'fids.json'))]
    if ref_dir is None:
        ref_dir = os.path.join(root_dir, 'ref')

    _, axes = plt.subplots(len(models_dirs), 3,
                           figsize=(scale * 4, scale * 2 * len(models_dirs)), dpi=200)

    if len(models_dirs) == 1:
        axes = [axes]
    for ax, model_dir in zip(axes, models_dirs):
        model_name = os.path.basename(model_dir)
        ax[0].set_title(model_name)
        inspect_random_fid_evaluation(model_dir, ref_dir, ax, ref_penalty=ref_penalty)


def std_mrrd(rand_models_dir, ref_model_dir):
    fids_random, pathes_random = load_fids_from_dir(rand_models_dir, 'fids_random')
    fids_original, pathes_original = load_fids_from_dir(ref_model_dir, 'fids_original')

    log_std = np.mean(np.log(fids_random[:, :-1]).std(axis=0))
    fids_random_summarized = fids_random.mean(axis=0)

    if fids_random_summarized.shape != fids_original.shape:
        return None, None
    mrrd = mean_shift(fids_original, fids_random_summarized)
    return log_std, mrrd


def inspect_model_activations(model_ref, model_rand, dataset):
    sample = next(iter(dataset)).cuda()
    original_out = model_ref(sample)[0]
    random_out = model_rand(sample)[0]

    plt.hist(original_out.flatten().cpu(), bins=30, color='Red', alpha=0.4, density=True);
    plt.hist(random_out.flatten().cpu(), bins=30, color='Green', alpha=0.4, density=True);


def sorted_samples(root_dir, dataset, is_random):
    fids, pathes = load_fids_from_dir(root_dir, 'fids_random' if is_random else 'fids_original')
    names = [os.path.basename(p) for p in pathes] + ['ref']
    if is_random:
        fids = fids.mean(axis=0)
    samples_loaders = [
        DataLoader(UnannotatedDataset(p), batch_size=9, shuffle=True) for p in pathes] + \
        [make_dataset_loader(dataset, 9)]
    imgs = []
    for sampler in samples_loaders:
        imgs.append(to_image(make_grid(next(iter(sampler)), nrow=3, pad_value=1)))

    permutation = sort_by_values(fids)
    print(sort_by(permutation, names))
    print(sort_by(permutation, fids))

    return sort_by(permutation, imgs), sort_by(permutation, names)