import numpy as np
import torch
from tqdm import tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, linear_rampup, sigmoid_rampup, linear_rampdown
import sys
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model1, model2, model_ema1, model_ema2, train_criterion1, train_criterion2, metrics, optimizer1, optimizer2, config, 
                 data_loader1, data_loader2,
                 valid_data_loader=None,
                 test_data_loader=None,
                 lr_scheduler1=None, lr_scheduler2=None,
                 len_epoch=None, val_criterion=None,
                 model_ema1_copy=None, model_ema2_copy=None):
        super().__init__(model1, model2, model_ema1, model_ema2, train_criterion1, train_criterion2, 
                         metrics, optimizer1, optimizer2, config, val_criterion, model_ema1_copy, model_ema2_copy)
        self.config = config.config
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader1)
        else:
            # iteration-based training
            self.data_loader1 = inf_loop(data_loader1)
            self.data_loader2 = inf_loop(data_loader2)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader

        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler1 = lr_scheduler1
        self.lr_scheduler2 = lr_scheduler2
        self.log_step = int(np.sqrt(self.data_loader1.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
        

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch, model, model_ema, model_ema2, data_loader, train_criterion, optimizer, lr_scheduler, device = 'cpu', queue = None):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        model.train()
        model_ema.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        total_metrics_ema = np.zeros(len(self.metrics))

        if hasattr(data_loader.dataset, 'num_raw_example'):
            num_examp = data_loader.dataset.num_raw_example
        else:
            num_examp = len(data_loader.dataset)

        local_step = 0



        with tqdm(data_loader) as progress:
            for batch_idx, (data, target, indexs, _) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')

                data_original = data
                target_original = target

                target = torch.zeros(len(target), self.config['num_classes']).scatter_(1, target.view(-1,1), 1)  
                data, target, target_original = data.to(device), target.float().to(device), target_original.to(device)
                
                data, target, mixup_l, mix_index = self._mixup_data(data, target,  alpha = self.config['mixup_alpha'], device = device)
                
                output = model(data)

                data_original = data_original.to(device)
                output_original  = model_ema2(data_original)
                output_original = output_original.data.detach()
                train_criterion.update_hist(epoch, output_original, indexs.numpy().tolist(), mix_index = mix_index, mixup_l = mixup_l)
                
                local_step += 1
                loss, probs = train_criterion(self.global_step + local_step, output, target)
                
                optimizer.zero_grad()
                loss.backward() 

                
                optimizer.step()
                
                self.update_ema_variables(model, model_ema, self.global_step + local_step, self.config['ema_alpha'])

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss', loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, target.argmax(dim=1))
                if output_original is not None:
                    total_metrics_ema += self._eval_metrics(output_original, target.argmax(dim=1))


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))

                if batch_idx == self.len_epoch:
                    break

        if hasattr(data_loader, 'run'):
            data_loader.run()


        log = {
            'global step': self.global_step,
            'local_step': local_step,
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'metrics_ema': (total_metrics_ema / self.len_epoch).tolist(),
            'learning rate': lr_scheduler.get_lr()
        }


        if lr_scheduler is not None:
            lr_scheduler.step()

        if queue is None:
            return log
        else:
            queue.put(log)


    def _valid_epoch(self, epoch, model1, model2, device = 'cpu', queue = None):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        model1.eval()
        model2.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, target, _, _) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, target = data.to(device), target.to(device)
                    
                    output1 = model1(data)
                    output2 = model2(data)

                    output = 0.5*(output1 + output2)

                    loss = self.val_criterion(output, target)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.writer.add_scalar('loss', loss.item())
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, target)
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # #add histogram of model parameters to the tensorboard
        # for name, p in model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        if queue is None:
            return {
                'val_loss': total_val_loss / len(self.valid_data_loader),
                'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
            }
        else:
            queue.put({
                'val_loss': total_val_loss / len(self.valid_data_loader),
                'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
            })

    def _test_epoch(self, epoch, model1, model2, device = 'cpu', queue = None):
        """
        Test after training an epoch

        :return: A log that contains information about test

        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        model1.eval()
        model2.eval()

        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, target,indexs,_) in enumerate(progress):
                    progress.set_description_str(f'Test epoch {epoch}')
                    data, target = data.to(device), target.to(device)

                    output1 = model1(data)
                    output2 = model2(data)
                    
                    output = 0.5*(output1 + output2)
                    loss = self.val_criterion(output, target)
                    self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                    self.writer.add_scalar('loss', loss.item())
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, target)
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            

        #add histogram of model parameters to the tensorboard
        for name, p in model1.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        if queue is None:
            return {
                'test_loss': total_test_loss / len(self.test_data_loader),
                'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
            }
        else:
            queue.put({
                'test_loss': total_test_loss / len(self.test_data_loader),
                'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
            })


    def _warmup_epoch(self, epoch, model, data_loader, optimizer, train_criterion, lr_scheduler, device = 'cpu', queue = None):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        model.train()

        with tqdm(data_loader) as progress:
            for batch_idx, (data, target, indexs , _) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')

                data, target = data.to(device), target.long().to(device)
                optimizer.zero_grad()
                output = model(data)
                out_prob = output.data.detach()
                
                train_criterion.update_hist(epoch, out_prob ,indexs.cpu().detach().numpy().tolist())

                loss = torch.nn.functional.cross_entropy(output, target)
        
                loss.backward() 
                optimizer.step()

                # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                # self.writer.add_scalar('loss', loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, target)


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))
                

                if batch_idx == self.len_epoch:
                    break

        log = {
            'loss': total_loss / self.len_epoch,
            'noise detection rate' : 0.0,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': lr_scheduler.get_lr()
        }
        if queue is None:
            return log
        else:
            queue.put(log)

    def eval_train(self, epoch, model_ema2, train_criterion):
        #model.eval()
        num_samples = args.num_batches*args.batch_size
        losses = torch.zeros(num_samples)
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()  
                output0  = model_ema2(inputs)
                output0 = output0.data.detach()
                output1, output2, output3 = None, None, None
                train_criterion.update_hist(epoch, output0, output1, output2, output3, indexs.numpy().tolist(),mix_index = mix_index, mixup_l = mixup_l)


    def update_ema_variables(self, model, model_ema, global_step, alpha_=0.997):
        # Use the true average until the exponential average is more correct
        if alpha_ == 0:
            ema_param.data = param.data
        else:
            if self.config['ema_update']:
                alpha = sigmoid_rampup(global_step + 1, self.config['ema_step'])*alpha_
            else:
                alpha = min(1 - 1 / (global_step + 1), alpha_)
            for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader1, 'n_samples'):
            current = batch_idx * self.data_loader1.batch_size
            total = self.data_loader1.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _mixup_data(self, x, y, alpha=1.0,  device = 'cpu'):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1-lam)
            batch_size = x.size()[0]
            mix_index = torch.randperm(batch_size).to(device)

            mixed_x = lam * x + (1 - lam) * x[mix_index, :]#
            mixed_target = lam * y + (1 - lam) * y[mix_index, :]


            return mixed_x, mixed_target, lam, mix_index
        else:
            lam = 1
            return x, y, lam, ...


    def _mixup_criterion(self, pred, y_a, y_b, lam, *args):
        loss_a, prob_a, entropy_a= self.train_criterion(pred, y_a, *args)
        loss_b, porb_b, entropy_b = self.train_criterion(pred, y_b, *args)
        return lam * loss_a + (1 - lam) * loss_b, lam * prob_a + (1-lam) * porb_b, lam * entropy_a + (1-lam) * entropy_b
