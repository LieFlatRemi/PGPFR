from .base import Base
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import os, os.path as osp
import importlib
import sys
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import copy
import torchmetrics
import shutil

import model_defs
import optimizers as optimizer_defs
import losses as loss_defs
import utils
from .helpers import *
# from .proto import *
from functools import reduce
from operator import mul

now_gpu = 0

class UnifiedPromptAdapterTuning(Base):
    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        super(UnifiedPromptAdapterTuning, self).__init__(cfg, cfg_data, args, is_train, is_distributed, n_gpus)
        self.loss_func = nn.CrossEntropyLoss().to(self.args.gpu)
        
        # 初始化类索引到任务索引的映射
        self.class_index_to_task_map = {}

    def train(self, n_trial):
        print(f"Using GPU = {self.args.gpu} with (batch_size, workers) = ({self.cfg.batch_size}, {self.cfg.workers})")
        torch.cuda.set_device(self.args.gpu)

        self.cfg.num_total_classes = self.cfg_data.get_n_classes(self.args.split_type)
        # Load model
        self.model = model_defs.get_model(edict({'n_classes': self.cfg.num_total_classes, **self.cfg.model}))

        # Class mapping vars
        c = 0

        self.cfg.class_mapping = {}

        label_to_name = self.cfg_data.label_to_name[self.args.split_type]
        self.cfg.label_to_name_mapped = {}
        # Run tasks
        for current_t_index in range(self.cfg.increm.max_task + 1):
            train_name = str(current_t_index)
            print('======================', train_name, '=======================')
            # Set variables depending on the task
            if current_t_index == 1:
                # Task 1: 使用Task 0的类别，不增加新类
                total_epochs_task = self.cfg.total_epochs_incremental_task
                self.cfg.total_epochs = self.cfg.total_epochs_incremental_task
                self.valid_out_dim = self.cfg.increm.first_split_size  # 保持与Task 0相同
                self.known_classes = 0  # 从0开始加载，确保加载的是Task 0的类
                self.add_classes = self.cfg.increm.first_split_size  # 加载Task 0的所有类
            elif current_t_index > 1:
                # Task 2+: 正常的持续学习，增加新类
                total_epochs_task = self.cfg.total_epochs_incremental_task
                self.cfg.total_epochs = self.cfg.total_epochs_incremental_task
                self.known_classes = self.valid_out_dim
                self.add_classes = self.cfg.increm.other_split_size
                self.valid_out_dim += self.cfg.increm.other_split_size
            else:
                # Task 0: 训练特征提取器和分类头
                total_epochs_task = self.cfg.total_epochs
                self.valid_out_dim = self.cfg.increm.first_split_size
                self.known_classes = 0
                self.add_classes = self.valid_out_dim


            # Load best checkpoint if desired. Otherwise, continue training from last checkpoint
            if current_t_index == 1 and self.cfg.increm.load_best_checkpoint_train:
                model_path = utils.get_best_model_path(osp.join(self.args.log_dir, f"task_{current_t_index - 1}"))
                assert model_path is not None, f"Model checkpoint not found in the log directory {self.args.log_dir}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                epoch = checkpoint['epoch']
                utils.load_state_dict_single(checkpoint['state_dict'], self.model)
                print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                del checkpoint

            model_defs.print_n_params(self.model)
            best_measure_info = utils.init_best_measure_info('acc', 'accuracy')
            log_dir_task = osp.join(self.args.log_dir, f"task_{train_name}")

            # load dataset for task
            self.train_dataset = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'),
                                         'Dataset')('train', self.args.split_type, self.cfg_data,
                                                    self.cfg.transforms['train'],
                                                    self.add_classes, self.known_classes,
                                                    rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)

            self.val_dataset = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'),
                                       'Dataset')('val', self.args.split_type, self.cfg_data,
                                                  self.cfg.transforms['val'],
                                                  self.add_classes, self.known_classes,
                                                  rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)

            print(f"Training classes: {self.train_dataset.keep_class_l}")

            # Class and label mapping
            # Task 1: 不更新映射，使用Task 0的映射
            if current_t_index != 1:
                for k in self.train_dataset.keep_class_l:
                    self.cfg.class_mapping[str(k)] = c
                    c += 1
                for prev_class, new_class in self.cfg.class_mapping.items():
                    self.cfg.label_to_name_mapped[str(new_class)] = label_to_name[int(prev_class)]

            # 更新类索引到任务索引的映射map
            if current_t_index > 0:
                for k in self.train_dataset.keep_class_l:
                    cil_index = self.cfg.class_mapping[str(k)]
                    self.class_index_to_task_map[cil_index] = current_t_index

            if current_t_index == 0 and self.cfg.increm.load_pretrained_task0:
                # Load pretrained model for task 0
                print("Loading pretrained model for task 0")
                self.valid_out_dim = self.cfg.increm.first_split_size
                # Create log dir if it does not exist
                if not osp.exists(osp.join(log_dir_task, 'checkpoints')):
                    os.makedirs(osp.join(log_dir_task, 'checkpoints'))
                # Copy checkpoint to log dir
                pretrained_checkpoint_path = osp.join('/ogr_cmu/models', self.args.dataset, self.cfg.model.name,
                                                      f"trial_{n_trial + 1}", 'checkpoints', 'model_best.pth.tar')
                assert osp.exists(
                    pretrained_checkpoint_path), f"Pretrained checkpoint not found in {pretrained_checkpoint_path}"
                shutil.copy(pretrained_checkpoint_path, osp.join(log_dir_task, 'checkpoints', 'model_best.pth.tar'))

                model_path = utils.get_best_model_path(osp.join(self.args.log_dir, f"task_0"))
                assert model_path is not None, f"Model checkpoint not found in the log directory {self.args.log_dir}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                epoch = checkpoint['epoch']
                utils.load_state_dict_single(checkpoint['state_dict'], self.model)
                self.model.cuda(self.args.gpu)

            else:
                self.train_sampler = None
                self.val_sampler = None

                # Append coreset samples to the train/val datasets if memory > 0
                self.train_dataset.append_coreset(self.coreset_train, self.ic, only=False)

                g = torch.Generator()
                g.manual_seed(3407)

                self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                                               shuffle=(self.train_sampler is None),
                                               num_workers=self.cfg.workers, pin_memory=True,
                                               sampler=self.train_sampler, drop_last=True if self.n_gpus > 1 else False,
                                               worker_init_fn=utils.seed_worker, generator=g)

                self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,
                                             shuffle=(self.val_sampler is None),
                                             num_workers=self.cfg.workers, pin_memory=True, sampler=self.val_sampler,
                                             drop_last=True if self.n_gpus > 1 else False,
                                             worker_init_fn=utils.seed_worker, generator=g)

                # Generate inverted samples
                if self.gen_inverted_samples:
                    self.generate_inverted_samples(self.previous_teacher, self.cfg.increm.learner.n_samples_per_class,
                                                   self.cfg.increm.learner.inversion_batch_size, self.cfg.batch_size,
                                                   log_dir_task, self.args.log_dir)

                # 扩展Prompt Pool (Task 2+才扩展，Task 1不扩展)
                if current_t_index > 1 and hasattr(self.model, 'prompt'):
                    self.model.prompt.update_prompt()

                # 扩展余弦分类器 (Task 2+才扩展，Task 1不扩展)
                if current_t_index > 1:
                    self.model.update_fc(self.cfg.increm.other_split_size)

                # Modify the base LR if current_task_index > 0
                if current_t_index > 0:
                    self.cfg.optimizer.lr = self.cfg.optimizer.lr_incremental_task
                    if not self.cfg.optimizer.include_scheduler and 'scheduler' in self.cfg.optimizer:
                        del self.cfg.optimizer.scheduler
                        del self.cfg.optimizer['scheduler']
                        self.cfg.step_per_epoch = False
                        self.cfg.step_per_batch = False
                        print("Scheduler deleted")
                self.optimizer, self.scheduler = optimizer_defs.get_optimizer_scheduler(self.model,
                                                                                        edict({**self.cfg.optimizer,
                                                                                               'total_epochs': total_epochs_task,
                                                                                               'n_steps_per_epoch': len(
                                                                                                   self.train_loader)}))

                self.criteria = loss_defs.get_losses(self.cfg.loss, self.valid_out_dim)

                resume_checkpoint_path = utils.get_last_checkpoint_path(log_dir_task)
                if resume_checkpoint_path:
                    print(f"=> loading checkpoint {resume_checkpoint_path}")
                    checkpoint = torch.load(resume_checkpoint_path, map_location=torch.device('cpu'))
                    start_epoch = checkpoint['epoch'] + 1
                    if start_epoch >= total_epochs_task:
                        print(f"Start epoch {start_epoch} is greater than total epochs {total_epochs_task}")
                        # sys.exit()
                    # Safely load state dicts
                    self.model.load_state_dict(checkpoint['state_dict']['model'])
                    if self.optimizer is not None and 'optimizer' in checkpoint['state_dict']:
                        self.optimizer.load_state_dict(checkpoint['state_dict']['optimizer'])
                    if self.scheduler is not None and 'scheduler' in checkpoint['state_dict']:
                        self.scheduler.load_state_dict(checkpoint['state_dict']['scheduler'])
                    print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                    del checkpoint

                else:
                    start_epoch = 1
                    print("=> no checkpoint found for resuming.")

                # Task 1+: 冻结特征提取器
                if current_t_index > 0 and self.cfg.increm.freeze_feature_extractor:
                    print("Freezing feature extractor...")
                    self.freeze_model(feature_extractor=True)

                # Freeze the weights for the previous classes in the classification layer if desired (from the second task onwards)
                if current_t_index > 0 and self.cfg.increm.freeze_classifier:
                    # Copy the weights and biases of the final linear layer
                    self.prev_weights = torch.empty_like(self.model.final.weight.data).copy_(
                        self.model.final.weight.data)
                    self.prev_bias = torch.empty_like(self.model.final.bias.data).copy_(self.model.final.bias.data)

                # transfer models
                self.model.cuda(self.args.gpu)

                # transfer optimizers and schedulers
                optimizer_defs.optimizer_to_cuda(self.optimizer, self.args.gpu)
                optimizer_defs.scheduler_to_cuda(self.scheduler, self.args.gpu)

                if self.args.gpu == now_gpu:
                    train_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'train'))
                    val_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'val'))

                    # epoch, train, val bars
                    print('Printing progress info for GPU 0 only ...')
                    ebar = tqdm(total=total_epochs_task - start_epoch + 1, leave=True, desc='epoch',
                                dynamic_ncols=False)
                    tbar = tqdm(total=len(self.train_loader), leave=True, desc='train', dynamic_ncols=False)
                    vbar = tqdm(total=len(self.val_loader), leave=True, desc='val', dynamic_ncols=False)

                step_per_epoch = False
                if 'scheduler' in self.cfg.optimizer:
                    if 'step_per_epoch' in self.cfg.optimizer.scheduler:
                        step_per_epoch = self.cfg.optimizer.scheduler.step_per_epoch


                # if self.cfg.increm.two_stage:
                # add: 两阶段训练 先训练 prompt 再训练 adater
                # task 0 反正也不需要adapter，直接正常走下面这段即可
                print("first freeze adapter")
                for param in self.model.spatial_adapter.parameters():
                    param.requires_grad = False
                for param in self.model.temporal_adapter.parameters():
                    param.requires_grad = False
                for epoch in range(start_epoch, total_epochs_task + 1):
                    torch.cuda.empty_cache()

                    self.train_epoch(tbar if self.args.gpu == now_gpu else None, epoch,
                                     train_logger if self.args.gpu == now_gpu else None, current_t_index)

                    measures = self.validate_epoch(vbar if self.args.gpu == now_gpu else None, epoch,
                                                   val_logger if self.args.gpu == now_gpu else None, current_t_index)

                    if self.args.gpu == now_gpu:
                        is_best = best_measure_info.func(measures[best_measure_info.tag], best_measure_info.val)
                        if is_best:
                            best_measure_info.val = measures[best_measure_info.tag]

                        train_logger.flush()
                        val_logger.flush()

                        if (epoch % self.args.save_epoch_freq == 0) and (self.args.gpu == now_gpu):
                            # save model
                            state_dict = utils.get_state_dict_single(self.model, self.optimizer, self.scheduler,
                                                                     self.is_distributed)

                            utils.save_checkpoint(log_dir_task,
                                                  {
                                                      'epoch': epoch,
                                                      'state_dict': state_dict,
                                                      'best_measure_tag': best_measure_info.tag,
                                                      'best_measure': best_measure_info.val,
                                                  },
                                                  epoch,
                                                  save_last_only=self.args.save_last_only,
                                                  is_best=is_best,
                                                  )

                    if step_per_epoch:
                        optimizer_defs.step_scheduler(self.scheduler)

                    if self.args.gpu == now_gpu:
                        ebar.update()
                        ebar.set_postfix(dict(epoch=epoch))

                if current_t_index > 0:
                    print("next freeze prompt")
                    for param in self.model.spatial_adapter.parameters():
                        param.requires_grad = True
                    for param in self.model.temporal_adapter.parameters():
                        param.requires_grad = True
                    for param in self.model.prompt.parameters():
                        param.requires_grad = False
                    for epoch in range(start_epoch, total_epochs_task + 1):
                        torch.cuda.empty_cache()

                        self.train_epoch(tbar if self.args.gpu == now_gpu else None, epoch,
                                         train_logger if self.args.gpu == now_gpu else None, current_t_index)

                        measures = self.validate_epoch(vbar if self.args.gpu == now_gpu else None, epoch,
                                                       val_logger if self.args.gpu == now_gpu else None,
                                                       current_t_index)

                        if self.args.gpu == now_gpu:
                            is_best = best_measure_info.func(measures[best_measure_info.tag], best_measure_info.val)
                            if is_best:
                                best_measure_info.val = measures[best_measure_info.tag]

                            train_logger.flush()
                            val_logger.flush()

                            if (epoch % self.args.save_epoch_freq == 0) and (self.args.gpu == now_gpu):
                                # save model
                                state_dict = utils.get_state_dict_single(self.model, self.optimizer, self.scheduler,
                                                                         self.is_distributed)

                                utils.save_checkpoint(log_dir_task,
                                                      {
                                                          'epoch': epoch,
                                                          'state_dict': state_dict,
                                                          'best_measure_tag': best_measure_info.tag,
                                                          'best_measure': best_measure_info.val,
                                                      },
                                                      epoch,
                                                      save_last_only=self.args.save_last_only,
                                                      is_best=is_best,
                                                      )

                        if step_per_epoch:
                            optimizer_defs.step_scheduler(self.scheduler)

                        if self.args.gpu == now_gpu:
                            ebar.update()
                            ebar.set_postfix(dict(epoch=epoch))

                    for param in self.model.prompt.parameters():
                        param.requires_grad = True


            self.last_valid_out_dim = self.valid_out_dim

            # set to eval mode
            self.model.eval()

            if self.args.gpu == now_gpu and not (current_t_index == 0 and self.cfg.increm.load_pretrained_task0):
                ebar.close()
                tbar.close()
                vbar.close()
                train_logger.close()
                val_logger.close()

        # save config
        if self.args.gpu == now_gpu:
            # Save config edict object
            utils.stdio.save_pickle(osp.join(self.args.log_dir, 'config.pkl'), self.cfg)

    def train_epoch(self, tbar, epoch, train_logger, current_t_index):

        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })

        # Class to save epoch metrics
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        n_batches = len(self.train_loader)

        # set to train mode
        self.model.train()

        # set epochs
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.train_sampler.epoch + 1)

        if self.args.gpu == now_gpu:
            tbar.reset(total=n_batches)
            tbar.refresh()

        step_per_batch = False
        if 'scheduler' in self.cfg.optimizer:
            if 'step_per_batch' in self.cfg.optimizer.scheduler:
                step_per_batch = self.cfg.optimizer.scheduler.step_per_batch

        iter_loader = iter(self.train_loader)
        bi = 1
        while bi <= n_batches:
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data['pts'], data['label']

            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            # output = self.model(pts)[:, :self.valid_out_dim]

            output, reduce_sim = self.model(x=pts, cur_task=current_t_index)
            output = output[:, :self.valid_out_dim]

            # trick：设置前面任务的logit为-inf
            # if current_t_index > 1:
            #     output[:, :self.valid_out_dim - self.add_classes] = -float('inf')

            loss = self.loss_func(output, target)

            prompt_loss = 0.0

            if current_t_index > 0:
                prompt_loss = reduce_sim.sum()
            # print('prompt loss : {} \t loss: {}'.format(prompt_loss, loss))
            loss = loss - (self.cfg.model.prompt.loss_coeff * prompt_loss)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            # # 检查参数更新
            # if bi == 1 and epoch == 1:
            #     self.check_params_update()

            if current_t_index > 0 and self.cfg.increm.freeze_classifier:
                # Restore the weights and biases for previous classes
                self.model.final.weight.data[:self.last_valid_out_dim] = self.prev_weights[:self.last_valid_out_dim]
                self.model.final.bias.data[:self.last_valid_out_dim] = self.prev_bias[:self.last_valid_out_dim]

            train_acc = acc_meter(output, target) * 100

            if step_per_batch:
                optimizer_defs.step_scheduler(self.scheduler)

            if self.args.gpu == now_gpu:
                tbar.update()
                tbar.set_postfix({
                    'it': bi,
                    'loss': loss.item(),
                    'train_acc': train_acc.item(),
                })
                tbar.refresh()

            bi += 1

        if self.args.gpu == now_gpu:
            acc_all = acc_meter.compute() * 100
            # hyperparam update
            train_logger.update(
                {'learning_rate': self.optimizer.param_groups[0]['lr']},
                step=epoch, prefix="stepwise")

            # loss update
            train_logger.update(
                {ltype: lmeter.avg for ltype, lmeter in losses.items()},
                step=epoch, prefix="loss")

            # measures update
            train_logger.update({
                'mean': acc_all,
            }, step=epoch, prefix="acc")

            acc_meter.reset()
            train_logger.flush()

    @torch.no_grad()
    def validate_epoch(self, vbar, epoch, val_logger, cur_task=0):

        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })

        # Class to save epoch metrics
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)

        # set to eval mode
        self.model.eval()

        if self.args.gpu == 0:
            vbar.reset(total=len(self.val_loader))
            vbar.refresh()

        n_batches = len(self.val_loader)
        iter_loader = iter(self.val_loader)
        bi = 1

        while bi <= n_batches:
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data['pts'], data['label']
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            # output = self.model(pts)[:, :self.valid_out_dim]
            output, reduce_sim = self.model(pts, cur_task=cur_task, train_mode=-1)

            # loss_tensors = []
            # for lname in self.criteria:
            #     lfunc = self.criteria[lname].func
            #     lweight = self.criteria[lname].weight
            #     lval = lfunc(output, target)
            #     losses[lname].update(lval.item(), output.size(0))
            #     loss_tensors.append(lweight * lval)
            #
            # loss = sum(loss_tensors)
            loss = self.loss_func(output, target)

            prompt_loss = 0.0
            if reduce_sim is not None:
                prompt_loss = reduce_sim.sum()
            loss = loss - (self.cfg.model.prompt.loss_coeff * prompt_loss)

            val_acc = acc_meter(output, target) * 100

            if self.args.gpu == now_gpu:
                vbar.update()
                vbar.set_postfix({
                    'it': bi,
                    'loss': loss.item(),
                    'val_acc': val_acc.item(),
                })
                vbar.refresh()

            bi += 1

        if self.args.gpu == now_gpu:
            acc_all = acc_meter.compute() * 100

            # loss update
            val_logger.update(
                {ltype: lmeter.avg for ltype, lmeter in losses.items()},
                step=epoch, prefix="loss")

            # measures update
            val_logger.update({
                'mean': acc_all,
            }, step=epoch, prefix="acc")

            acc_meter.reset()
            val_logger.flush()

            return_values = {ltype: lmeter.avg for ltype, lmeter in losses.items()}
            return_values['acc'] = acc_all

            return return_values

    def evaluate(self, n_trial):
        is_testval = (self.args.train == 0)
        mode = 'testval' if is_testval else 'test'

        # load config
        cfg = utils.stdio.load_pickle(osp.join(self.args.log_dir, 'config.pkl'))

        self.model = model_defs.get_model(edict({'n_classes': self.cfg.increm.first_split_size, **cfg.model}))

        # Test each task
        for current_t_index in range(cfg.increm.max_task + 1):
            if current_t_index == 1:
                self.valid_out_dim = self.cfg.increm.first_split_size
            elif current_t_index > 1:
                self.valid_out_dim += self.cfg.increm.other_split_size
            else:
                self.valid_out_dim = self.cfg.increm.first_split_size
            # print name
            cfg.test_name = str(current_t_index)
            log_dir_task = osp.join(self.args.log_dir, f"task_{cfg.test_name}")
            print('======================', cfg.test_name, '=======================')

            if current_t_index > 1:
                # 更新分类器
                self.model.update_fc(self.cfg.increm.other_split_size)
                # 如果prompt是逐增的，增加prompt
                if hasattr(self.model, 'prompt'):
                    self.model.prompt.update_prompt()

            model_defs.print_n_params(self.model)
            for test_mode in ['local', 'global', 'old', 'new']:
                if current_t_index > 1:
                    if test_mode == 'local':
                        self.known_classes = self.valid_out_dim - self.cfg.increm.other_split_size
                        self.add_classes = self.cfg.increm.other_split_size
                    elif test_mode == 'global':
                        self.known_classes = 0
                        self.add_classes = self.valid_out_dim
                    elif test_mode == 'old':
                        self.known_classes = 0
                        self.add_classes = self.cfg.increm.first_split_size
                    elif test_mode == 'new':
                        self.known_classes = self.cfg.increm.first_split_size
                        self.add_classes = self.valid_out_dim - self.cfg.increm.first_split_size
                else:
                    self.known_classes = 0
                    self.add_classes = self.valid_out_dim
                cfg.test_mode = test_mode
                print('======================', cfg.test_mode, '=======================')
                # define dataset
                self.test_dataset = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'),
                                            'Dataset')(mode, self.args.split_type, self.cfg_data,
                                                       self.cfg.transforms[mode],
                                                       self.add_classes, self.known_classes,
                                                       rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)

                g = torch.Generator()
                g.manual_seed(3407)

                self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                              num_workers=cfg.workers, pin_memory=True, sampler=None, drop_last=False,
                                              worker_init_fn=utils.seed_worker, generator=g)

                print(f"Testing classes: {self.test_dataset.keep_class_l}")
                # load checkpoint
                cfg.increm.load_best_checkpoint_test = self.cfg.increm.load_best_checkpoint_test
                if self.cfg.increm.load_best_checkpoint_test and current_t_index == 0:
                    model_path = utils.get_best_model_path(log_dir_task)
                else:
                    model_path = utils.get_last_checkpoint_path(log_dir_task)
                assert model_path is not None, \
                    f"Model checkpoint not found in the log directory {log_dir_task}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                utils.load_state_dict_single(checkpoint['state_dict'], self.model)
                print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                del checkpoint

                # transfer models
                self.model.cuda(self.args.gpu)

                # evaluate
                self.cfg = cfg
                self.evaluate_task()

        if cfg.increm.max_task > 1:
            self.save_accuracies_task()

    @torch.no_grad()
    def evaluate_task(self):
        # Class to save epoch metrics
        acc_meter = utils.Meter(self.valid_out_dim, self.cfg.label_to_name_mapped)
        acc_meter_torchmetrics = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(
            self.args.gpu)

        # set to eval mode
        self.model.eval()

        tbar = tqdm(total=len(self.test_loader), leave=True, desc='test', dynamic_ncols=False)
        tbar.refresh()

        n_batches = len(self.test_loader)

        iter_loader = iter(self.test_loader)
        bi = 1

        while bi <= n_batches:
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data['pts'], data['label']
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            x_task = [None] * len(target)
            # 获取每个样本对应的task id
            for i, target_class in enumerate(target):
                x_task[i] = self.class_index_to_task_map[target[i].cpu().item()]

            # output = self.model(pts)[:, :self.valid_out_dim]
            output, reduce_sim = self.model(pts, cur_task=-1, train_mode=-1, x_task=x_task)
            output = output[:, :self.valid_out_dim]

            acc_meter.update(output, target)
            test_acc_ = acc_meter_torchmetrics(output, target) * 100

            tbar.update()
            tbar.set_postfix({
                'it': bi,
                'test_acc': test_acc_.item(),
            })
            tbar.refresh()

            bi += 1

        tbar.close()

        test_folder = osp.join(self.args.log_dir, f"task_{self.cfg.test_name}", 'test')
        if not osp.exists(test_folder):
            os.makedirs(test_folder)

        if self.args.save_conf_mat:
            conf_mat = acc_meter.conf_matrix.squeeze().cpu().numpy()

            utils.save_conf_mat_image(
                conf_mat,
                self.cfg.label_to_name_mapped,
                osp.join(test_folder, f"conf_mat_{self.cfg.test_mode}.png"),
            )

        acc_all = acc_meter.accuracies()
        acc_all_torchmetrics = acc_meter_torchmetrics.compute() * 100
        acc_meter_torchmetrics.reset()
        with open(osp.join(test_folder, f"test_metrics_{self.cfg.test_mode}.json"), 'w') as f:
            json.dump({'Acc': acc_all, 'Acc_torchmetrics': acc_all_torchmetrics.item()}, f, indent=4)

    @torch.no_grad()
    def check_params_update(self):
        print("\n" + "=" * 80)
        print("检查参与更新的参数:")
        print("=" * 80)

        trainable_params = []
        frozen_params = []
        params_with_grad = []
        params_without_grad = []

        for name, param in self.model.named_parameters():
            info = {
                'name': name,
                'shape': tuple(param.shape),
                'requires_grad': param.requires_grad,
                'has_grad': param.grad is not None,
                'grad_norm': param.grad.norm().item() if param.grad is not None else 0.0
            }

            if param.requires_grad:
                trainable_params.append(info)
                if param.grad is not None:
                    params_with_grad.append(info)
                else:
                    params_without_grad.append(info)
            else:
                frozen_params.append(info)

        print(f"\n可训练参数总数: {len(trainable_params)}")
        print(f"有梯度的参数: {len(params_with_grad)}")
        print(f"无梯度的参数: {len(params_without_grad)}")
        print(f"冻结的参数: {len(frozen_params)}")

        print("\n--- 可训练且有梯度的参数 ---")
        for info in sorted(params_with_grad, key=lambda x: x['grad_norm'], reverse=True)[:20]:  # 只显示前20个
            print(f"{info['name']:60s} | shape: {str(info['shape']):30s} | grad_norm: {info['grad_norm']:.6f}")

        if len(params_with_grad) > 20:
            print(f"... 还有 {len(params_with_grad) - 20} 个参数有梯度")

        print("\n--- 可训练但无梯度的参数（可能有问题！）---")
        if params_without_grad:
            for info in params_without_grad:
                print(f"{info['name']:60s} | shape: {str(info['shape']):30s}")
        else:
            print("无")

        print("\n--- 关键组件参数检查 ---")
        key_components = ['prompt', 'cls', 'query', 'initial', 'spatial_att', 'temporal_att']
        for component in key_components:
            component_params = [info for info in trainable_params if component in info['name']]
            if component_params:
                print(f"\n{component.upper()} 组件:")
                total_grad_norm = sum(info['grad_norm'] for info in component_params if info['has_grad'])
                print(f"  参数数量: {len(component_params)}, 总梯度范数: {total_grad_norm:.6f}")
                for info in component_params[:5]:  # 只显示前5个
                    status = "✓有梯度" if info['has_grad'] else "✗无梯度"
                    print(f"    {info['name']:50s} | {status} | grad_norm: {info['grad_norm']:.6f}")

        print("\n" + "=" * 80 + "\n")

    def freeze_model(self, feature_extractor=False, classifier=False):
        if feature_extractor:
            # Freeze initial layer
            for param in self.model.initial.parameters():
                param.requires_grad = False
            # Freeze spatial_att
            for param in self.model.spatial_att.parameters():
                param.requires_grad = False
            # Freeze temporal_att
            for param in self.model.temporal_att.parameters():
                param.requires_grad = False

            if hasattr(self.model, 'prompt_query') and hasattr(self.model.prompt_query, 'query'):
                for param in self.model.prompt_query.initial.parameters():
                    param.requires_grad = False
                for param in self.model.prompt_query.spatial_att.parameters():
                    param.requires_grad = False
                for param in self.model.prompt_query.temporal_att.parameters():
                    param.requires_grad = False
            print('freeze feature extractor')
        if classifier:
            # Freeze classifier
            for param in self.model.final.parameters():
                param.requires_grad = False