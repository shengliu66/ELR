from typing import TypeVar, List, Tuple
import torch
from tqdm import tqdm
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import numpy as np
from slacker import Slacker

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model1, model2, model_ema1, model_ema2, train_criterion1, train_criterion2, metrics, optimizer1, optimizer2, config, val_criterion):
        self.config = config
        if self.config['visdom']['server'] is not None:
            self.viz = vis_visdom(self.config)
        else:
            self.viz = None

        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        if self.config['slack']['slack_token']:
            with open(self.config['slack']['slack_token'], 'r') as f:
                slack_api_token = f.read().replace('\n','')
            self.slack = Slacker(slack_api_token)
            try:
                if self.slack.api.test().successful:
                    print(f"Successfully connected to {self.slack.team.info().body['team']['name']}.")
                    message = 'Experiment Name: {:s}'.format(self.config['name'])
                    self._report_stats_slack(text = message, channel = 'training_report')
                else:
                    print('Try Again!')
            except:
                
                self.slack = None
        else:
            self.slack = None

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model1 = model1.to(self.device)
        self.model2 = model2.to(self.device)
        if model_ema1 is not None:
            self.model_ema1 = model_ema1.to(self.device)
        else:
            self.model_ema1 = None

        if model_ema2 is not None:
            self.model_ema2 = model_ema2.to(self.device)
        else:
            self.model_ema2 = None

        if len(device_ids) > 1:
            self.model1 = torch.nn.DataParallel(model1, device_ids=device_ids)
            self.model2 = torch.nn.DataParallel(model2, device_ids=device_ids)
            if self.model_ema1 is not None:
                self.model_ema1  = torch.nn.DataParallel(model_ema1, device_ids=device_ids)
            if self.model_ema2 is not None:
                self.model_ema2  = torch.nn.DataParallel(model_ema2, device_ids=device_ids)
        
        if self.model_ema1 is not None:
            for param in self.model_ema1.parameters():
                param.detach_()

        if self.model_ema2 is not None:
            for param in self.model_ema2.parameters():
                param.detach_()

        self.train_criterion1 = train_criterion1.to(self.device)
        self.train_criterion2 = train_criterion2.to(self.device)
        
        
        self.val_criterion = val_criterion
        self.metrics = metrics

        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.global_step = 0

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epochs number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        out_pred_all = []
        out_grad_all = []
        out_pred_test_all = []
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
            if epoch <= self.config['trainer']['warmup']:
                result1 = self._warmup_epoch(epoch, self.model1, self.data_loader1, self.optimizer1, self.train_criterion1, self.lr_scheduler1)
                result2 = self._warmup_epoch(epoch, self.model2, self.data_loader2, self.optimizer2, self.train_criterion2, self.lr_scheduler2)
            else:
                result1, out_pred1 = self._train_epoch(epoch, self.model1, self.model_ema1, self.model_ema2, self.data_loader1, self.train_criterion1, self.optimizer1, self.lr_scheduler1)
                result2, out_pred2 = self._train_epoch(epoch, self.model2, self.model_ema2, self.model_ema1, self.data_loader2, self.train_criterion2, self.optimizer2, self.lr_scheduler2)
                self.global_step += self.local_step

                if self.do_validation:
                    val_log = self._valid_epoch(epoch)
                    result1.update(val_log)
                    result2.update(val_log)
                if self.do_test:
                    test_log, test_meta = self._test_epoch(epoch)
                    result1.update(test_log)
                    result2.update(test_log)
                else: 
                    test_meta = [0,0]
                out_pred1.append(test_meta)

                out_pred_all.append(out_pred1[0])
                out_grad_all.append(out_pred1[2])
                out_pred_test_all.append(out_pred1[-1][0])

            

                

                

            if self.viz is not None: 
                self.viz.draw(epoch, result1)
                # self.viz.plot_heatmap(self.train_criterion.weight_ret)
            # self.viz.plot_hist(self.train_criterion.Acc_mat[0], 0)
            # self.viz.plot_hist(self.train_criterion.Acc_mat[1], 1)
            #for kk in range(10):
            #    self.viz.plot_hist(self.train_criterion.pred[:,kk].cpu().squeeze(), kk)
            
            
            
            # mask = np.ones(self.train_criterion.pred_target.cpu().squeeze().shape,dtype=bool)
            # try:
            #     mask[self.data_loader.dataset.noise_indx] = False
            # except:
            #     pass
            # data_vis = self.train_criterion.pred_target.data.cpu().squeeze().numpy()
            # data_vis = (data_vis - data_vis.min())/(data_vis.max() - data_vis.min() + 1e-6)
            # self.viz.plot_hist(data_vis[~mask],10) #GT Noisy
            # self.viz.plot_hist(data_vis[mask],11) #GT Clean
            # print('Model think gt is wrong: ',(data_vis[mask]<0.01).nonzero())
            # print('Model think noise is right: ', (data_vis[~mask]>0.98).nonzero())
            

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result1.items():
                if key == 'metrics':
                    log.update({'Net1' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                    log.update({'Net2' + mtr.__name__: result2[key][i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'test_metrics':
                    log.update({'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log['Net1'+key] = value
                    log['Net2'+key] = result2[key]

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if (self.slack is not None) and (epoch > self.config['trainer']['warmup']):
                message = 'Epoch: {:d} Net1 Training Loss: {:.4f} Net2 Training Loss: {:.4f} Validation Acc: {:.4f} Test Acc: {:.4f}'.format(epoch,log['Net1loss'],log['Net2loss'],log['val_my_metric'],log['test_my_metric'])
                self._report_stats_slack(text = message, channel = 'mmt_training_report')

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
                if epoch > self.config['trainer']['warmup']:
                    self._save_output(out_pred_all, out_pred1[1], out_grad_all, out_pred_test_all, out_pred1[-1][1])

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model1).__name__

        if hasattr(self.train_criterion1, 'pred_hist') and hasattr(self.train_criterion2, 'pred_hist'):
            pred_hist1 = self.train_criterion1.pred_hist
            pred_hist2 = self.train_criterion2.pred_hist
        else:
            pred_hist= None
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict1': self.model1.state_dict(),
            'state_dict2': self.model2.state_dict(),
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict(),
            'monitor_best': self.mnt_best,
            'pred_hist1': pred_hist1,
            'pred_hist2': pred_hist2
            #'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _save_output(self, out_pred_all, target_all, grad_all, out_pred_test_all, target_test_all):
        filename = str(self.checkpoint_dir / 'all-outputs.npy')
        filename_tar = str(self.checkpoint_dir / 'all-targets.npy')
        filename_test = str(self.checkpoint_dir / 'all-outputs-test.npy')
        filename_tar_test = str(self.checkpoint_dir / 'all-targets-test.npy')
        filename_grad = str(self.checkpoint_dir / 'all-grads.npy')
        np.save(filename, out_pred_all)
        np.save(filename_tar, target_all)
        np.save(filename_test, out_pred_test_all)
        np.save(filename_tar_test, target_test_all)
        np.save(filename_grad, grad_all)


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _report_stats_slack(self, text, channel):
        """Report training stats"""
        r = self.slack.chat.post_message(channel=channel, text=text,
                                    username='Training Report',
                                    icon_emoji=':clipboard:')

        if r.successful:
            return True
        else:
            return r.error


class vis_visdom:
    def __init__(self, config):
        import visdom
        viz = visdom.Visdom(port=config['visdom']['port'], server='http://'+config['visdom']['server'])
        self.viz = viz
        viz.env = config['name']
        self.loss_plot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Losses',
                title='Train & Val Losses',
                legend=['Train-Loss', 'Val-Loss']
            )
        )

        self.noise_rate = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, )).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Confidence',
                title='Train'
            )
        )

        self.eval_plot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 6)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Accuracy',
                title='Train & Val Accuracies',
                legend=['trainTop1','valTop1','testTop1','trainTop3','valTop3','testTop3']
            )
        )
        # self.M_plot = viz.heatmap(
        #     X=np.outer(np.arange(1, 10), np.arange(1, 10)),
        #     opts=dict(
        #         #columnnames=['CN','MCI','AD'],
        #         #rownames=['CN','MCI','AD'],
        #         title='Cost Matrix',
        #         colormap='Electric',
        #         )
        #     )
        # self.hist_plot = dict()
        # for i in range(12):
        #     self.hist_plot[i] = viz.histogram(
        #         X=np.random.rand(10000),
        #         opts=dict(numbins=50, xtickmin=-0.001, xtickmax=1.01, ytickmin=0,
        #             title='Logit on class' + str(i)
        #             )
        #         )
    def draw(self, epoch, result):

        if 'test_metrics' in result:
            test_acc = result['test_metrics']
        else:
            test_acc = [0,0]
        if 'val_metrics' in result:
            val_acc = result['val_metrics']
        else:
            val_acc = [0,0]
        if 'val_loss' in result:
            val_loss = result['val_loss']
        else:
            val_loss = 0
        self.viz.line(
                X=torch.ones((1, 2)).cpu() * epoch-1,
                Y=torch.Tensor([result['loss'], val_loss]).unsqueeze(0).cpu(),
                win=self.loss_plot,
                update='append'
            )

        self.viz.line(
                X=torch.ones((1, )).cpu() * epoch-1,
                Y=torch.Tensor([result['noise detection rate']]).unsqueeze(0).cpu(),
                win=self.noise_rate,
                update='append'
            )
        self.viz.line(
                X=torch.ones((1, 6)).cpu() * epoch-1,
                Y=torch.Tensor([result['metrics'][0], val_acc[0], test_acc[0], result['metrics'][1], val_acc[1], test_acc[1]]).unsqueeze(0).cpu(),
                win=self.eval_plot,
                update='append'
            )
        
    def plot_heatmap(self,M):
        self.viz.heatmap(
            X=M,
            win=self.M_plot
            )
    def plot_hist(self, x, win_i):
        self.viz.histogram(
            X = x,
            win=self.hist_plot[win_i],
            opts=dict(numbins=50, ytickmin=0, ytickmax=20,
                    title='Logit on class' + str(win_i)
                    )
            )