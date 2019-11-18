from torch_tools.utils import wrap_with_tqdm


def init_with(model, initializer, min_dim=2):
    for param in model.parameters():
        if len(param.shape) >= min_dim:
            initializer(param)


def fit_bn_to_dataset(model, dataset, random_init, steps_for_fit=50, no_bias=False):
    if random_init is not None:
        init_with(model, random_init)

    is_training = model.training
    model.cuda()
    model.train()

    print(no_bias)
    if no_bias:
        for name, param in model.named_parameters():
            if 'bias' in name:
                param *= 0

    for i, sample in wrap_with_tqdm(enumerate(dataset)):
        sample = sample.cuda()
        model(sample)
        if i >= steps_for_fit:
            break

    model.train(is_training)
