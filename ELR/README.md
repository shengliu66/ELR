# ELR
This is an official PyTorch implementation of ELR method proposed in [Early-Learning Regularization Prevents Memorization of Noisy Labels](https://arxiv.org/abs/2007.00151). 


## Usage
Train the network on the Symmmetric Noise CIFAR-10 dataset (noise rate = 0.8):

```
python train.py -c config_cifar10.json --percent 0.8
```
Train the network on the Asymmmetric Noise CIFAR-10 dataset (noise rate = 0.4):

```
python train.py -c config_cifar10_asym.json --percent 0.4 --asym 1
```

Train the network on the Asymmmetric Noise CIFAR-100 dataset (noise rate = 0.4):

```
python train.py -c config_cifar100.json --percent 0.4 --asym 1
```

The config files can be modified to adjust hyperparameters and optimization settings. 

## Results
### CIFAR10
<center>

| Method                 |  20%        |    40%      |   60%        |      80%    |    40% Asym |
| ---------------------- | ----------- | ----------- | -----------  | ----------- | ----------- |
| ELR                    | 91.16%      | 89.15%      |  86.12%      | 73.86%      |     90.12%  |
| ELR (cosine annealing) | 91.12%      | 91.43%      |  88.87%      | 80.69%      |    90.35%   |

### CIAFAR100

| Method                 |  20%        |    40%      |   60%        |      80%    |    40% Asym |
| ---------------------- | ----------- | ----------- | -----------  | ----------- | ----------- |
| ELR                    | 74.21%      | 68.28%      |  59.28%      | 29.78%      |    73.71%  |
| ELR (cosine annealing) | 74.68%      | 68.43%      |  60.05%      | 30.27%      |    73.96%   |

</center>

## References
- S. Liu, J. Niles-Weed, N. Razavian and C. Fernandez-Granda "Early-Learning Regularization Prevents Memorization of Noisy Labels", 2020
