# RandomFID

Original pytorch implementation of anonymous CVPR2020 submission _Random-FID: a Simple yet Effective Alternative to Frechet Inception Distance for Non-Imagenet Domains_.

Random-FID is a random networks based modification of the standard FID that can be used on any domain.

# Pretrained models and gans samples

MNIST generators ouputs (without MNIST-5 and test MNIST):
https://drive.google.com/file/d/1Y0PYHH-DCQUtx6ypZjJoJ_dhQdjvyMiC/view?usp=sharing

Pretrained CIFAR-10 and MNIST models:
https://drive.google.com/open?id=1aBOAETCFkTXK7WDUpG6p0qhwxOSGr0-d

# How to run?

1. Make sure there are pretrained models in ./pretrained; Put all gans directories samples to some root folder and specify it in data.py
2. Go to ```./lib```. Run evaluation: ```run_for_all.sh```

# License
Code for FID computation is based on
https://github.com/mseitzer/pytorch-fid
~