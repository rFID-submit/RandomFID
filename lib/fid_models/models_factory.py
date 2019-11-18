from torch import nn
from fid_models.model_activations import model_activations
from fid_models.inception import make_random_inception
from fid_models.lenet import make_random_lenet


INITIALIZERS = {
    'kaiming': nn.init.kaiming_normal_,
    'uniform': nn.init.uniform_,
    'xavier': nn.init.xavier_normal_,
    None: None
}


def make_random_model(model_name, original_dataloader,
                      random_init=None, steps_for_fit=50, activation=None, pretrained=False,
                      no_bias=False, linear=False):
    kwargs = {
        'random_init': INITIALIZERS[random_init],
        'steps_for_fit': steps_for_fit,
        'no_bias': no_bias
    }
    if model_name == 'inception':
        random_model = make_random_inception(original_dataloader, **kwargs)
    elif model_name == 'LeNet':
        random_model = make_random_lenet(original_dataloader, pretrained=pretrained, **kwargs)
    else:
        random_model = model_activations(model_name, original_dataloader,
                                         activations=activation,
                                         pretrained=pretrained,
                                         do_linearize=linear, **kwargs)
    return random_model.eval().cuda()