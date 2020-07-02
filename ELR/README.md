# ELR
This is an official PyTorch implementation of ELR method proposed in [Early-Learning Regularization Prevents Memorization of Noisy Labels](). 


## Usage
Train the network on the Symmmetric Noise CIFAR-10 dataset (noise rate = 0.8):

```
python train.py -c config_cifar10.json --percent 0.8
```
to train and relabel the dataset.


Train the network on the Asymmmetric Noise CIFAR-10 dataset (noise rate = 0.4):

```
python train.py -c config_cifar100.json --percent 0.4 --asym 1
```

The config files can be modified to adjust hyperparameters and optimization settings. 


## References
- S. Liu, J. N. Weed, N. Razavian and C. F. Granda "Joint Optimization Framework for Learning with Noisy Labels", 2020