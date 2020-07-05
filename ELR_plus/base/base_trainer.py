from typing import TypeVar, List, Tuple
import torch
from tqdm import tqdm
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import numpy as np


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model1, model2, model_ema1, model_ema2, train_criterion1, 
                train_criterion2, metrics, optimizer1, optimizer2, config, val_criterion,
                model_ema1_copy, model_ema2_copy):
        self.config = config.config

        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])


        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])

        if len(self.device_ids) > 1:
            print('Using Multi-Processing!')

        self.model1 = model1.to(self.device+str(self.device_ids[0]))
        self.model2 = model2.to(self.device+str(self.device_ids[-1]))

        if model_ema1 is not None:
            self.model_ema1 = model_ema1.to(self.device+str(self.device_ids[0]))
            self.model_ema2_copy = model_ema2_copy.to(self.device+str(self.device_ids[0]))
        else:
            self.model_ema1 = None
            self.model_ema2_copy = None

        if model_ema2 is not None:
            self.model_ema2 = model_ema2.to(self.device+str(self.device_ids[-1]))
            self.model_ema1_copy = model_ema1_copy.to(self.device+str(self.device_ids[-1]))
        else:
            self.model_ema2 = None
            self.model_ema1_copy = None
        
        if self.model_ema1 is not None:
            for param in self.model_ema1.parameters():
                param.detach_()

            for param in self.model_ema2_copy.parameters():
                param.detach_()

        if self.model_ema2 is not None:
            for param in self.model_ema2.parameters():
                param.detach_()

            for param in self.model_ema1_copy.parameters():
                param.detach_()

        
        self.train_criterion1 = train_criterion1.to(self.device+str(self.device_ids[0]))
        self.train_criterion2 = train_criterion2.to(self.device+str(self.device_ids[-1]))

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

        if len(self.device_ids) > 1:
            import torch.multiprocessing as mp
            mp.set_start_method('spawn', force =True)
            
        not_improved_count = 0

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
            if epoch <= self.config['trainer']['warmup']:
                if len(self.device_ids) > 1:
                    q1 = mp.Queue()
                    q2 = mp.Queue()
                    p1 = mp.Process(target=self._warmup_epoch, args=(epoch, self.model1, self.data_loader1, self.optimizer1, self.train_criterion1, self.lr_scheduler1, self.device+str(self.device_ids[0]), q1 ))
                    p2 = mp.Process(target=self._warmup_epoch, args=(epoch, self.model2, self.data_loader2, self.optimizer2, self.train_criterion2, self.lr_scheduler2, self.device+str(self.device_ids[-1]), q2))
                    p1.start() 
                    p2.start()
                    result1 = q1.get()
                    result2 = q2.get()
                    p1.join()
                    p2.join()
                else:
                    result1 = self._warmup_epoch(epoch, self.model1, self.data_loader1, self.optimizer1, self.train_criterion1, self.lr_scheduler1, self.device+str(self.device_ids[0]))
                    result2 = self._warmup_epoch(epoch, self.model2, self.data_loader2, self.optimizer2, self.train_criterion2, self.lr_scheduler2, self.device+str(self.device_ids[-1]))
                
                if len(self.device_ids) > 1:
                    self.model_ema1_copy.load_state_dict(self.model_ema1.state_dict())
                    self.model_ema2_copy.load_state_dict(self.model_ema2.state_dict())
                    if self.do_validation:
                        q1 = mp.Queue()
                        p1 = mp.Process(target=self._valid_epoch, args=(epoch, self.model1, self.model_ema2_copy, self.device+str(self.device_ids[0]),q1))
                        
                    if self.do_test:
                        q2 = mp.Queue()
                        p2 = mp.Process(target=self._test_epoch, args=(epoch, self.model1, self.model_ema2_copy, self.device+str(self.device_ids[0]),q2))
                        p1.start()
                        p2.start()
                        val_log = q1.get()
                        test_log, test_meta = q2.get()
                        result1.update(val_log)
                        result2.update(val_log)
                        result1.update(test_log)
                        result2.update(test_log)
                    p1.join()
                    p2.join()
                else: 
                    if self.do_validation:
                        val_log = self._valid_epoch(epoch, self.model1, self.model2, self.device+str(self.device_ids[0]))
                        result1.update(val_log)
                        result2.update(val_log)
                    if self.do_test:
                        test_log, test_meta = self._test_epoch(epoch, self.model1, self.model2, self.device+str(self.device_ids[0]))
                        result1.update(test_log)
                        result2.update(test_log)
                    else:
                        test_meta = [0,0]

            else:
                if len(self.device_ids) > 1:
                    q1 = mp.Queue()
                    q2 = mp.Queue()
                    p1 = mp.Process(target=self._train_epoch, args=(epoch, self.model1, self.model_ema1, self.model_ema2_copy, self.data_loader1, self.train_criterion1, self.optimizer1, self.lr_scheduler1, self.device+str(self.device_ids[0]), q1 ))
                    p2 = mp.Process(target=self._train_epoch, args=(epoch, self.model2, self.model_ema2, self.model_ema1_copy, self.data_loader2, self.train_criterion2, self.optimizer2, self.lr_scheduler2, self.device+str(self.device_ids[-1]), q2 ))
                    p1.start() 
                    p2.start()
                    result1 = q1.get()
                    result2 = q2.get()
                    p1.join()
                    p2.join()
                else:
                    result1 = self._train_epoch(epoch, self.model1, self.model_ema1, self.model_ema2, self.data_loader1, self.train_criterion1, self.optimizer1, self.lr_scheduler1, self.device+str(self.device_ids[0]))
                    result2 = self._train_epoch(epoch, self.model2, self.model_ema2, self.model_ema1, self.data_loader2, self.train_criterion2, self.optimizer2, self.lr_scheduler2, self.device+str(self.device_ids[-1]))


                self.global_step += result1['local_step']
                if len(self.device_ids) > 1:
                    self.model_ema1_copy.load_state_dict(self.model_ema1.state_dict())
                    self.model_ema2_copy.load_state_dict(self.model_ema2.state_dict())
                    if self.do_validation:
                        q1 = mp.Queue()
                        p1 = mp.Process(target=self._valid_epoch, args=(epoch, self.model1, self.model_ema2_copy, self.device+str(self.device_ids[0]),q1))
                        
                    if self.do_test:
                        q2 = mp.Queue()
                        p2 = mp.Process(target=self._test_epoch, args=(epoch, self.model1, self.model_ema2_copy, self.device+str(self.device_ids[0]),q2))
                        p1.start()
                        p2.start()
                        val_log = q1.get()
                        test_log = q2.get()
                        result1.update(val_log)
                        result2.update(val_log)
                        result1.update(test_log)
                        result2.update(test_log)
                    p1.join()
                    p2.join()
                else: 
                    if self.do_validation:
                        val_log = self._valid_epoch(epoch, self.model1, self.model2, self.device+str(self.device_ids[0]))
                        result1.update(val_log)
                        result2.update(val_log)
                    if self.do_test:
                        test_log = self._test_epoch(epoch, self.model1, self.model2, self.device+str(self.device_ids[0]))
                        result1.update(test_log)
                        result2.update(test_log)    

            

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
        device = 'cuda:'#torch.device('cuda:' if n_gpu_use > 0 else 'cpu')
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

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict1': self.model1.state_dict(),
            'state_dict2': self.model2.state_dict(),
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict(),
            'monitor_best': self.mnt_best
            #'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth at: {} ...".format(best_path))



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
        if checkpoint['config']['arch'] != self.config['arch1']:
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

