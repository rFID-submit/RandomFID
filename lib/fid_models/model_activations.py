from torch import nn
from torchvision import models
from fid_models.initialization import fit_bn_to_dataset
from fid_models.inception_v3_cut import InceptionV3Cut
from linearization import linearize


def save_hook(module, input, output):
    setattr(module, 'output', output)


class Activations(nn.Module):
    def __init__(self, model, activations, postprocessing=None):
        super(Activations, self).__init__()
        if (not isinstance(activations, list)) and (not isinstance(activations, tuple)):
            activations = [activations]

        self.model = model
        self.activations = {}
        for name, module in model.named_modules():
            if name in activations:
                self.activations[name] = module
                module.register_forward_hook(save_hook)

        self.postprocessing = postprocessing

    def layers_count(self):
        return len(self.activations)

    def forward(self, *input):
        self.model.forward(*input)
        if self.postprocessing is not None:
            for activation in self.activations.values():
                activation.output = self.postprocessing(activation.output)

        out = {a[0]: a[1].output.view([input[0].shape[0], -1]) for a in self.activations.items()}
        if self.layers_count() == 1:
            return next(iter(out.values()))
        else:
            return out


def activaion_name(model_name):
    if 'resnet' in model_name:
        return 'avgpool'
    elif 'vgg' in model_name:
        return 'classifier.0'
    elif model_name == 'densenet121':
        return 'features.denseblock4.denselayer16'
    elif model_name == 'mobilenet_v2':
        return 'features.18.0'
    elif 'inception_v3' in model_name:
        return 'model.Mixed_7c'


def model_activations(target_model, dataset=None, steps_for_fit=50, random_init=None,
                      activations=None, pretrained=False, do_linearize=False, no_bias=False):
    if target_model in ['inception_v3_mnist', 'inception_v3_cifar10']:
        model_class = InceptionV3Cut
    else:
        model_class = getattr(models, target_model)

    model = model_class(pretrained=pretrained).cuda()

    if not pretrained:
        if do_linearize:
            linearize(model)
        fit_bn_to_dataset(model, dataset, random_init, steps_for_fit, no_bias)
    model.eval()

    postprocessing = None
    if target_model == 'mobilenet_v2':
        postprocessing = lambda x: x.mean([2, 3])
    elif target_model == 'densenet121' or 'inception' in target_model:
        postprocessing = lambda x: nn.functional.adaptive_avg_pool2d(x, (1, 1))

    activations = activaion_name(target_model) if activations is None else activations
    return Activations(model, activations, postprocessing)
