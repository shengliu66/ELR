import argparse
import collections
import sys
import requests
import socket
import torch
import mlflow
import mlflow.pytorch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from collections import OrderedDict
import random


def log_params(conf: OrderedDict, parent_key: str = None):
    for key, value in conf.items():
        if parent_key is not None:
            combined_key = f'{parent_key}-{key}'
        else:
            combined_key = key

        if not isinstance(value, OrderedDict):
            mlflow.log_param(combined_key, value)
        else:
            log_params(value, combined_key)


def main(config: ConfigParser):


    logger = config.get_logger('train')
    logger.info(config.config)

    # setup data_loader instances
    data_loader1 = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=config['data_loader']['args']['validation_split'],
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'] 
    )

    data_loader2 = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size2'],
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=config['data_loader']['args']['validation_split'],
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'] 
    )


    valid_data_loader = data_loader1.split_validation()

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()

    # build model architecture
    model1 = config.initialize('arch1', module_arch)
    model_ema1 = config.initialize('arch1', module_arch)
    model_ema1_copy = config.initialize('arch1', module_arch)
    model2 = config.initialize('arch2', module_arch)
    model_ema2 = config.initialize('arch2', module_arch)
    model_ema2_copy = config.initialize('arch2', module_arch)
    

    # get function handles of loss and metrics
    device_id = list(range(min(torch.cuda.device_count(), config['n_gpu'])))

    if hasattr(data_loader1.dataset, 'num_raw_example') and hasattr(data_loader2.dataset, 'num_raw_example'):
        num_examp1 = data_loader1.dataset.num_raw_example
        num_examp2 = data_loader2.dataset.num_raw_example
    else:
        num_examp1 = len(data_loader1.dataset)
        num_examp2 = len(data_loader2.dataset)

    train_loss1 = getattr(module_loss, config['train_loss']['type'])(num_examp=num_examp1, num_classes=config['num_classes'],
                                                            device = 'cuda:'+ str(device_id[0]), config = config.config, alpha=config['train_loss']['args']['alpha'])
    train_loss2 = getattr(module_loss, config['train_loss']['type'])(num_examp=num_examp2, num_classes=config['num_classes'],
                                                            device = 'cuda:'+str(device_id[-1]), config = config.config, alpha=config['train_loss']['args']['alpha'])

    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
    trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

    optimizer1 = config.initialize('optimizer1', torch.optim, [{'params': trainable_params1}])
    optimizer2 = config.initialize('optimizer2', torch.optim, [{'params': trainable_params2}])

    lr_scheduler1 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer1)
    lr_scheduler2 = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer2)

    trainer = Trainer(model1, model2, model_ema1, model_ema2, train_loss1, train_loss2, 
                      metrics, 
                      optimizer1, optimizer2,
                      config=config,
                      data_loader1=data_loader1,
                      data_loader2=data_loader2,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler1=lr_scheduler1,
                      lr_scheduler2=lr_scheduler2,
                      val_criterion=val_loss,
                      model_ema1_copy = model_ema1_copy,
                      model_ema2_copy = model_ema2_copy)

    trainer.train()
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--alpha', '--alpha'], type=float, target=('train_loss', 'args', 'alpha')),
        CustomArgs(['--lambda', '--lambda'], type=float, target=('train_loss', 'args', 'lambda')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--malpha', '--mixup_alpha'], type=float, target=('mixup_alpha',)),
        CustomArgs(['--ealpha', '--ema_alpha'], type=float, target=('ema_alpha',)),
        CustomArgs(['--nb', '--num_batches'], type=float, target=('data_loader', 'args', 'num_batches')),
        CustomArgs(['--warm', '--warmup'], type=int, target=('trainer', 'warmup')),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--wc1', '--weight_decay1'], type=float, target=('optimizer1','weight_decay')),
        CustomArgs(['--wc2', '--weight_decay2'], type=float, target=('optimizer2','weight_decay')),
        CustomArgs(['--estep', '--ema_step'], type=float, target=('ema_step',)),

    ]
    config = ConfigParser.get_instance(args, options)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    main(config)