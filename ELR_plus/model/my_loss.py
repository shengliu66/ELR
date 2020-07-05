import torch.nn.functional as F
import torch
import numpy as np
from parse_config import ConfigParser
import torch.nn as nn
from torch.autograd import Variable
import math

def update_label(probs):
    max_idx = torch.argmax(probs, 0, keepdim=True)
    one_hot = torch.cuda.FloatTensor(probs.shape)
    one_hot.zero_()
    one_hot.scatter_(0, max_idx, 1)
    return one_hot



class our_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, alpha=0.5, lamb=0.7):
        super(our_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.pred_hist = (torch.ones(num_examp, self.num_classes)*1.0/self.num_classes).cuda()
        self.p = torch.ones(self.num_classes).cuda() / self.num_classes if self.USE_CUDA else torch.ones(self.num_classes) / self.num_classes
        self.alpha = alpha
        self.lamb = lamb



    def forward(self, output, target, epoch, index):
        y_true = make_one_hot(target, C=self.num_classes)
        y_pred = F.softmax(output,dim=1)
        config = ConfigParser.get_instance()
        y_true_1 = y_true
        y_pred_1 = y_pred#(y_pred**(1.0/3))/(y_pred**(1.0/3)).sum(dim=1,keepdim=True)

        y_true_2 = y_true
        y_pred_2 = y_pred#(y_pred**(1.0/3))/(y_pred**(1.0/3)).sum(dim=1,keepdim=True)

        y_pred_1 = torch.clamp(y_pred_1, 1e-3, 1.0)
        y_true_2 = torch.clamp(y_true_2, 1e-7, 1.0-1e-7)


        avg_probs = torch.mean(y_pred, dim=0)


        L_p = -torch.sum(torch.log(avg_probs) * self.p)

        pred_ = y_pred_2.data.detach()

        self.pred_hist[index] = (1-self.lamb) * self.pred_hist[index] + self.lamb * ((pred_**self.alpha)/(pred_**self.alpha).sum(dim=1,keepdim=True))

 
        weight =  (1-self.pred_hist[index])

        t = 3.0

        out = ((self.weight * y_pred_1)).sum(dim=1)       
        
        ce_loss = torch.mean(-torch.sum((y_true_1  ) * F.log_softmax(output, dim=1), dim = -1))
            
        mae_loss = (out.log()).mean()

        Entropy = - (F.softmax(output.data.detach(), dim=1) * F.log_softmax(output.data.detach(), dim=1)).sum(dim=1).mean()

        return config['alpha']*ce_loss + config['beta']*mae_loss+ config['gamma']*L_p , self.pred_hist[index].cpu(), Entropy


