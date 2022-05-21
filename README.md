<div align="center">   
  
# Early-Learning Regularization Prevents Memorization of Noisy Labels
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2007.00151-green)](https://arxiv.org/abs/2007.00151)

</div>

This repository is the official implementation of [Early-Learning Regularization Prevents Memorization of Noisy Labels](https://arxiv.org/abs/2007.00151) (NeurIPS 2020).

We propose a novel framework to perform classification via deep learning in the presence of **noisy annotations**. When trained on noisy labels, deep neural networks have been observed to first fit the training data with clean labels during an **early learning** phase, before eventually **memorizing** the examples with false labels. Our technique exploits the progress of the early learning phase via **regularization** to perform classification from noisy labels. There are two key elements to our approach. First, we leverage semi-supervised learning techniques to produce target probabilities based on the model outputs. Second, we design a regularization term that steers the model towards these targets, implicitly preventing memorization of the false labels. The resulting framework is shown to provide robustness to noisy annotations on several standard benchmarks and real-world datasets, where it achieves results comparable to the state of the art.

<p float="left" align="center">
<img src="images/illustration_of_ELR.png" width="800" /> 
<figcaption align="center">
These graphs show the results of training a ResNet-34 with a traditional cross entropy loss (top row) and our proposed method (bottom row) to perform classification on the CIFAR-10 dataset where 40% of the labels are flipped at random. The left column shows the fraction of examples with clean labels that are predicted correctly (green) and incorrectly (blue). The right column shows the fraction of examples with wrong labels that are predicted correctly (green), memorized (the prediction equals the wrong label, shown in red), and incorrectly predicted as neither the true nor the labeled class (blue). The model trained with cross entropy begins by learning to predict the true labels, even for many of the examples with wrong labels, but eventually memorizes the wrong labels. Our proposed method based on early-learning regularization prevents memorization, allowing the model to continue learning on the examples with clean labels to attain high accuracy on examples with both clean and wrong labels.
</figcaption>
</p>

<p float="left" align="center">
<img src="images/clean_label_simplexheatmap2.gif" width="400" /> 
<img src="images/false_label_simplexheatmap.gif" width="400" /> 
<figcaption align="center">
Learning path of sample with correct label (left) and sample with wrong label (right). Corners correspond to one-hot
vectors. Bright green represents model's prediction: when the example is wrongly labeled, the clean label is predicted at initialization, and then wrong label is predicted at the end of training. The model is trained with first 3 classes in CIFAR10.
</figcaption>
</p>

## Requirements
- This codebase is written for `python3`.
- To install necessary python packages, run `pip install -r requirements.txt`.


## Training
### Basics
- ELR loss is implemented in the file [`loss.py`](./ELR/model/loss.py)
- All functions used for training the basic version of our technique (**ELR**) can be found in the `ELR` folder.
- All functions used for training the more advanced version (**ELR+**) can be found in the `ELR_plus` folder.
- Experiments settings and configurations used for different datasets are in the corresponding config json files.
### Data
- Please download the data before running the code, add path to the downloaded data to `data_loader.args.data_dir` in the corresponding config file.
### Training
- Code for training ELR is in the following file: [`train.py`](./ELR/train.py), code for training ELR+ is in the following file: [`train.py`](./ELR_plus/train.py) 
```
usage: train.py [-c] [-r] [-d] [--lr learning_rate] [--bs batch_size] [--beta beta] [--lambda lambda] [--malpha mixup_alpha]
                [--percent percent] [--asym asym] [--ealpha ema_alpha]  [--name exp_name] 

  arguments:
    -c, --config                  config file path (default: None)
    -r, --resume                  path to latest checkpoint (default: None)
    -d, --device                  indices of GPUs to enable (default: all)     
  
  options:
    --lr learning_rate            learning rate (default value is the value in the config file)
    --bs batch_size               batch size (default value is the value in the config file)
    --beta beta                   temporal ensembling momentum beta for target estimation
    --lambda lambda               regularization coefficient
    --malpha mixup_alpha          mixup parameter alpha
    --percent percent             noise level (e.g. 0.4 for 40%)
    --asym asym                   asymmetric noise is used when set to True
    --ealpha ema_alpha            weight averaging momentum for target estimation
     --name exp_name              experiment name
```
Configuration file is **required** to be specified. Default option values, if not reset, will be the values in the configuration file. 
Examples for ELR and ELR+ are shown in the *readme.md* of `ELR` and `ELR_plus` subfolders respectively.

### Example
In order to use our proposed early learning regularization (ELR), you can simply replace your loss function by the following loss function. Usually, **lambda** which used to control the strength of the regularization term need to be tuned more carefully, and the value of **beta** is often quite robust (can be 0.7, 0.9 or 0.99, etc.)
```
class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, lambda = 3, beta=0.7):
         r"""Early Learning Regularization.
         Parameters
         * `num_examp` Total number of training examples.
         * `num_classes` Number of classes in the classification problem.
         * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
         * `beta` Temporal ensembling momentum for target estimation.
         """

        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        self.lambda = lambda
        

    def forward(self, index, output, label):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """

        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.lambda *elr_reg
        return  final_loss

```
## Identify Wrong Labels
- After finish training, obtain and compare the self.target of the ELR loss to original labels y
- the mislabeled examples are identified as those who has self.target != y


## License and Contributing
- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github. 

## Reference
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2007.00151).
```
@article{liu2020early,
  title={Early-Learning Regularization Prevents Memorization of Noisy Labels},
  author={Liu, Sheng and Niles-Weed, Jonathan and Razavian, Narges and Fernandez-Granda, Carlos},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
Similar early learning and memorization phenomenon is observed in semantic segmentations, a related paper addressing WSSS by adaptive correction: 
[Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](https://arxiv.org/abs/2110.03740) (CVPR2022 **Oral**).
## Contact
Please contact shengliu@nyu.edu if you have any question on the codes.
