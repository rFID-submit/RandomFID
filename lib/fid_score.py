#!/usr/bin/env python3
# code based on //github.com/mseitzer/pytorch-fid.git
import os
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch_tools.utils import wrap_with_tqdm


def get_activations(dataloader, model, cuda, verbose):
    model.eval()

    batch_size = dataloader.batch_size
    pred_arr = None

    samples_count = 0
    for i, images in wrap_with_tqdm(enumerate(dataloader),
                                    verbosity=verbose, total=len(dataloader)):
        if cuda:
            images = images.cuda()
        images = 0.5 * (images + 1.0)

        with torch.no_grad():
            pred = model(images)
        if isinstance(pred, list):
            pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if len(pred.shape) == 4:
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        features = pred.cpu().data.numpy().reshape(batch_size, -1)
        if pred_arr is None:
            pred_arr = np.empty([len(dataloader) * batch_size, features.shape[1]])

        pred_arr[i * batch_size: (i + 1) * batch_size] = features
        samples_count += batch_size

    pred_arr = pred_arr[:samples_count]

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, as_total_variation=False):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    print('shift component: {} | var component: {} | im part: {} (mean: {})'.format(
        diff.dot(diff),
        np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean,
        np.max(np.abs(covmean.imag)),
        np.mean(np.abs(covmean.imag))
    ))

    if as_total_variation:
        diff = 2.0 * (np.clip(mu1, 0, 1) - np.clip(mu2, 0, 1))
        return np.max(np.abs(diff))

    return diff.dot(diff) + (np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model, cuda, verbose):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataloader  : images tensors dataloader
    -- model       : Instance of inception model
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(dataloader, model, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_for_dataloaders(original_data_loader, dataloaders, cuda=True,
                                  verbose=False, model=None, debug_dir=None):
    print('calculating dataset statistics')
    m_orig, s_orig = \
        calculate_activation_statistics(original_data_loader, model, cuda, verbose)
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, 'm_orig'), m_orig)
        np.save(os.path.join(debug_dir, 's_orig'), s_orig)

    fids = []
    for i, gen in enumerate(dataloaders):
        print('generator: {} / {}'.format(i + 1, len(dataloaders)))
        m, s = calculate_activation_statistics(gen, model, cuda, verbose)
        fids.append(calculate_frechet_distance(m, s, m_orig, s_orig))
        if debug_dir is not None:
            np.save(os.path.join(debug_dir, 'm_{}'.format(i)), m)
            np.save(os.path.join(debug_dir, 's_{}'.format(i)), s)

    return fids
