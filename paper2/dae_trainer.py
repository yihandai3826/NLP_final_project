import numpy as np
import random
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from layers import LossWrapper, ScheduledOptim, ExponentialDecay
from logger import DAELogger
from config import device
import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DAETrainer(object):

    def __init__(self, hps):
        self.hps = hps

    def run_validation(self, epoch, mixpoet, tool, lr):
        logger = DAELogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.dae_valid_log_path)
        logger.set_rate('learning_rate', lr)
        logger.set_rate('teach_ratio', mixpoet.get_teach_ratio())

        for step in range(0, tool.valid_batch_num):
            batch = tool.valid_batches[step]
            batch_keys = batch[0].to(device)
            batch_poems = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            batch_lengths = batch[3].to(device)

            gen_loss, _ = self.run_step(mixpoet, None,
                batch_keys, batch_poems, batch_dec_inps, batch_lengths, True)
            logger.add_losses(gen_loss)

        logger.print_log(epoch)
        return gen_loss  # 返回验证损失

    def run_step(self, mixpoet, optimizer, keys, poems, dec_inps, lengths, valid=False):
        if not valid:
            optimizer.zero_grad()
        all_outs = mixpoet.dae_graph(keys, poems, dec_inps, lengths)

        gen_loss = self.losswrapper.cross_entropy_loss(all_outs, dec_inps)

        if not valid:
            gen_loss.backward()
            clip_grad_norm_(mixpoet.dae_parameters(), self.hps.clip_grad_norm)
            optimizer.step()

        return gen_loss.item(), all_outs

    def run_train(self, mixpoet, tool, optimizer, scheduler, logger):
        logger.set_start_time()

        for step in range(0, tool.train_batch_num):
            batch = tool.train_batches[step]
            batch_keys = batch[0].to(device)
            batch_poems = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            batch_lengths = batch[3].to(device)

            gen_loss, outs = self.run_step(mixpoet, optimizer,
                batch_keys, batch_poems, batch_dec_inps, batch_lengths)

            logger.add_losses(gen_loss)
            logger.set_rate("learning_rate", optimizer.param_groups[0]['lr'])
            if step % self.hps.dae_log_steps == 0:
                logger.set_end_time()
                utils.sample_dae(batch_keys, batch_poems, batch_dec_inps,
                    outs, self.hps.sample_num, tool)
                logger.print_log()
                logger.set_start_time()

    def train(self, mixpoet, tool):
        utils.print_parameter_list(mixpoet, mixpoet.dae_parameter_names())
        print("building data for dae...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.dae_batch_size, mode='dae')

        print("train batch num: %d" % (tool.train_batch_num))
        print("valid batch num: %d" % (tool.valid_batch_num))

        logger = DAELogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.dae_log_steps)
        logger.set_log_path(self.hps.dae_train_log_path)
        logger.set_rate('learning_rate', 0.0)
        logger.set_rate('teach_ratio', 1.0)

        opt = torch.optim.AdamW(mixpoet.dae_parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizer = ScheduledOptim(optimizer=opt, warmup_steps=self.hps.dae_warmup_steps,
            max_lr=self.hps.dae_max_lr, min_lr=self.hps.dae_min_lr)

        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

        mixpoet.train()
        self.losswrapper = LossWrapper(pad_idx=tool.get_PAD_ID(), sens_num=self.hps.sens_num,
            sen_len=self.hps.sen_len)

        tr_decay_tool = ExponentialDecay(self.hps.dae_burn_down_tr, self.hps.dae_decay_tr,
            self.hps.dae_min_tr)

        for epoch in range(1, self.hps.dae_epoches+1):
            self.run_train(mixpoet, tool, optimizer, scheduler, logger)

            if epoch % self.hps.dae_validate_epoches == 0:
                print("run validation...")
                mixpoet.eval()
                print("in training mode: %d" % (mixpoet.training))
                val_loss = self.run_validation(epoch, mixpoet, tool, optimizer.rate())
                scheduler.step(val_loss)  # 更新学习率
                mixpoet.train()
                print("validation Done: %d" % (mixpoet.training))

            if (self.hps.dae_save_epoches >= 1) and \
                (epoch % self.hps.dae_save_epoches) == 0:
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, mixpoet, prefix="dae")

            logger.add_epoch()
            print("teach forcing ratio decay...")
            mixpoet.set_teach_ratio(tr_decay_tool.do_step())
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())
            print("shuffle data...")
            tool.shuffle_train_data()