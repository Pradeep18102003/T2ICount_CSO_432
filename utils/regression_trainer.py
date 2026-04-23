from torch.utils.data import DataLoader, default_collate
import torch
import logging
from utils.helper import SaveHandler, AverageMeter
from utils.trainer import Trainer
from models.reg_model import Count
from datasets.dataset import ObjectCount
import numpy as np
import os
import time
import random
import shutil
import torch.nn as nn
from utils.ssim_loss import cal_avg_ms_ssim
from utils.tools import extract_patches, reassemble_patches
from torch.cuda.amp import autocast, GradScaler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    den = torch.stack(transposed_batch[1], 0)  # the number of points is not fixed, keep it as a list of tensor
    prompt = transposed_batch[2]
    prompt_attn_mask = torch.stack(transposed_batch[3], 0)
    img_attn_mask = torch.stack(transposed_batch[4], 0)
    return images, den, prompt, prompt_attn_mask, img_attn_mask


class Reg_Trainer(Trainer):
    def setup(self):
        args = self.args
        if args.seed != -1:
            setup_seed(args.seed)
            print('Random seed is set as {}'.format(args.seed))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_device(torch.cuda.current_device())
            self.device_count = torch.cuda.device_count()
            logging.info('Using {} gpus'.format(self.device_count))
        else:
            self.device = torch.device('cpu')
            self.device_count = 1
            logging.info('Using CPU')

        self.d_ratio = args.downsample_ratio

        self.datasets = {x: ObjectCount(args.data_dir,
                                        crop_size=args.crop_size,
                                        downsample_ratio=self.d_ratio,
                                        method=x,
                                        concat_size=args.concat_size) for x in ['train', 'val', 'test']}

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          collate_fn=(train_collate if x=='train' else default_collate),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val', 'test']}


        self.model = Count(args.config, args.sd_path,
                           unet_config={'base_size': self.args.crop_size,
                                        'max_attn_size': self.args.crop_size // self.d_ratio,
                                        'attn_selector': 'down_cross+up_cross'})
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW([
            {'params': self.model.unet.parameters(),
             'lr': args.lr * 0.1,
             'weight_decay': args.weight_decay * 0.1},
            {'params': self.model.decoder.parameters(),
             'lr': args.lr,
             'weight_decay': args.weight_decay}])
        
        self.start_epoch = 0    
        self.scaler = GradScaler()

        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                raise Exception('Not supported')

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_list = SaveHandler(num=args.max_num)

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            logging.info('-' * 50 + "Epoch:{}/{}".format(epoch, args.epochs - 1) + '-' * 50)
            self.epoch = epoch
            self.train_epoch()
            if self.epoch >= args.start_val and self.epoch % self.args.val_epoch == 0:
                self.val_epoch()
        self.save_final_model()

    def save_final_model(self):
        final_name = getattr(self.args, 'final_model_name', 'final_model.pth')
        final_model_path = os.path.join(self.save_dir, final_name)
        torch.save(self.model.state_dict(), final_model_path)
        logging.info(f"Saved final model to {final_model_path}")

        # Kaggle-specific convenience: persist a copy in /kaggle/working for easy download.
        kaggle_output_dir = '/kaggle/working'
        if os.path.isdir(kaggle_output_dir):
            kaggle_model_path = os.path.join(kaggle_output_dir, final_name)
            try:
                shutil.copy2(final_model_path, kaggle_model_path)
                logging.info(f"Copied final model to Kaggle output: {kaggle_model_path}")
            except Exception as e:
                logging.warning(f"Could not copy final model to {kaggle_model_path}: {e}")

    def train_epoch(self):
        epoch_reg_loss = AverageMeter()
        epoch_RRC1_loss = AverageMeter()
        epoch_RRC2_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()

        for step, (input, den_map, caption, prompt_attn_mask, img_attn_mask) in enumerate(
                self.dataloaders['train']):
            inputs = input.to(self.device)
            gt_den_maps = den_map.to(self.device) * 60
            gt_prompt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
            gt_img_attn_mask = img_attn_mask.to(self.device)
            self.model.set_train()
            with torch.set_grad_enabled(True), autocast():
                N = inputs.shape[0]
                pred_den, sim_x2, sim_x1, fused_cross_attn = self.model(inputs, caption, gt_prompt_attn_mask)
                fused_cross_attn_ = fused_cross_attn * gt_img_attn_mask
                AN = fused_cross_attn_ >= 0.3 
                reg_loss = get_reg_loss(pred_den, gt_den_maps, threshold=1e-3 * 60)
                P = gt_den_maps >= (1e-3 * 60)
                rrc_loss_stage1 = RRC_loss(sim_x2, AN, P)
                rrc_loss_stage2 = RRC_loss(sim_x1, AN, P)

                epoch_reg_loss.update(reg_loss.item(), N)
                epoch_RRC1_loss.update(rrc_loss_stage1.item(), N)
                epoch_RRC2_loss.update(rrc_loss_stage2.item(), N)
                loss = reg_loss + 0.01 * rrc_loss_stage1 + 0.01 * rrc_loss_stage2

                gt_counts = torch.sum(gt_den_maps.view(N, -1), dim=1).detach().cpu().numpy() / 60
                pred_counts = torch.sum(pred_den.view(N, -1), dim=1).detach().cpu().numpy() / 60
                diff = pred_counts - gt_counts
                epoch_mae.update(np.mean(np.abs(diff)).item(), N)
                epoch_mse.update(np.mean(diff * diff), N)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        logging.info(
            'Epoch {} Train, reg:{:.4f}, RRC_stage1:{:.4f}, RRC_stage2:{:.4f}, mae:{:.2f}, mse:{:.2f}, Cost: {:.1f} sec '
            .format(self.epoch, epoch_reg_loss.getAvg(), epoch_RRC1_loss.getAvg(), epoch_RRC2_loss.getAvg(), epoch_mae.getAvg(),
                    np.sqrt(epoch_mse.getAvg()), (time.time() - epoch_start)))

        if self.epoch % 5 == 0:
            model_state_dict = self.model.state_dict()
            save_path = os.path.join(self.save_dir, "{}_ckpt.tar".format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dict,
            }, save_path)
            self.save_list.append(save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.set_eval()
        epoch_res = []
        skipped_samples = 0
        pred_counts_epoch = []
        for inputs, gt_counts, captions, prompt_attn_mask, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            gt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
            cropped_imgs, num_h, num_w = extract_patches(inputs, patch_size=self.args.crop_size,
                                                         stride=self.args.stride)
            outputs = []
            with torch.set_grad_enabled(False):
                num_chunks = (cropped_imgs.size(0) + self.args.batch_size - 1) // self.args.batch_size
                for i in range(num_chunks):
                    start_idx = i * self.args.batch_size
                    end_idx = min((i + 1) * self.args.batch_size, cropped_imgs.size(0))
                    outputs_partial = self.model(cropped_imgs[start_idx:end_idx], captions * (end_idx - start_idx), gt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))[0]
                    outputs.append(outputs_partial)
                results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, inputs.size(2), inputs.size(3),
                                             patch_size=self.args.crop_size, stride=self.args.stride)
                if not torch.isfinite(results).all():
                    skipped_samples += 1
                    logging.warning(f"Val sample {name[0]} produced non-finite density map. Skipping this sample.")
                    continue
                pred_count = torch.sum(results).item() / 60
                res = gt_counts[0].item() - torch.sum(results).item() / 60
                if not np.isfinite(res):
                    skipped_samples += 1
                    logging.warning(f"Val sample {name[0]} produced non-finite residual {res}. Skipping this sample.")
                    continue
                pred_counts_epoch.append(pred_count)
                epoch_res.append(res)

        if len(epoch_res) == 0:
            logging.warning("All validation samples were non-finite after reassembly. Check stride/crop_size and loss stability.")
            mae = np.nan
            mse = np.nan
        else:
            epoch_res = np.array(epoch_res)
            mse = np.sqrt(np.mean(np.square(epoch_res)))
            mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Val, MAE: {:.2f}, MSE: {:.2f} Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))
        if len(pred_counts_epoch) > 0:
            logging.info('Epoch {} Val, avg_pred_count: {:.2f}'.format(self.epoch, float(np.mean(pred_counts_epoch))))
        if skipped_samples > 0:
            logging.info(f"Validation skipped {skipped_samples} samples due to non-finite predictions.")

        model_state_dict = self.model.state_dict()

        if (mae + mse) < (self.best_mae + self.best_mse):
            self.best_mae = mae
            self.best_mse = mse
            torch.save(model_state_dict, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.epoch)))
            logging.info("Save best model: MAE: {:.2f} MSE:{:.2f} model epoch {}".format(mae, mse, self.epoch))
            self.test_epoch()
        print("Best Result: MAE: {:.2f} MSE:{:.2f}".format(self.best_mae, self.best_mse))

    def test_epoch(self):
        epoch_start = time.time()
        self.model.set_eval()
        epoch_res = []
        skipped_samples = 0
        pred_counts_epoch = []
        for inputs, gt_counts, captions, prompt_attn_mask, name in self.dataloaders['test']:
            inputs = inputs.to(self.device)
            gt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
            cropped_imgs, num_h, num_w = extract_patches(inputs, patch_size=self.args.crop_size,
                                                         stride=self.args.stride)
            outputs = []
            with torch.set_grad_enabled(False):
                num_chunks = (cropped_imgs.size(0) + self.args.batch_size - 1) // self.args.batch_size
                for i in range(num_chunks):
                    start_idx = i * self.args.batch_size
                    end_idx = min((i + 1) * self.args.batch_size, cropped_imgs.size(0))
                    outputs_partial = self.model(cropped_imgs[start_idx:end_idx], captions * (end_idx - start_idx), gt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))[0]
                    outputs.append(outputs_partial)
                results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, inputs.size(2), inputs.size(3),
                                             patch_size=self.args.crop_size, stride=self.args.stride)
                if not torch.isfinite(results).all():
                    skipped_samples += 1
                    logging.warning(f"Test sample {name[0]} produced non-finite density map. Skipping this sample.")
                    continue
                pred_count = torch.sum(results).item() / 60
                res = gt_counts[0].item() - torch.sum(results).item() / 60
                if not np.isfinite(res):
                    skipped_samples += 1
                    logging.warning(f"Test sample {name[0]} produced non-finite residual {res}. Skipping this sample.")
                    continue
                pred_counts_epoch.append(pred_count)
                epoch_res.append(res)

        if len(epoch_res) == 0:
            logging.warning("All test samples were non-finite after reassembly. Check stride/crop_size and loss stability.")
            mae = np.nan
            mse = np.nan
        else:
            epoch_res = np.array(epoch_res)
            mse = np.sqrt(np.mean(np.square(epoch_res)))
            mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Test, MAE: {:.2f}, MSE: {:.2f} Cost {:.1f} sec'
                     .format(self.epoch, mae, mse, (time.time() - epoch_start)))
        if len(pred_counts_epoch) > 0:
            logging.info('Epoch {} Test, avg_pred_count: {:.2f}'.format(self.epoch, float(np.mean(pred_counts_epoch))))
        if skipped_samples > 0:
            logging.info(f"Test skipped {skipped_samples} samples due to non-finite predictions.")

def get_normalized_map(density_map):
    B, C, H, W = density_map.size()
    density_map = torch.clamp(density_map, min=0.0)
    mu_sum = density_map.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    mu_normed = density_map / torch.clamp(mu_sum, min=1e-6)
    return mu_normed



def get_reg_loss(pred, gt, threshold, level=3, window_size=3):
    mask = gt > threshold
    loss_ssim = cal_avg_ms_ssim(pred * mask, gt * mask, level=level,
                                window_size=window_size)
    mu_normed = get_normalized_map(pred)
    gt_mu_normed = get_normalized_map(gt)
    tv_loss = (nn.L1Loss(reduction='none')(mu_normed, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)
    return loss_ssim + 0.1 * tv_loss


def RRC_loss(simi, ambiguous_negative_map, positive_map):
    pos = (1 - simi) * positive_map
    neg = torch.clamp(simi, min=0) * (ambiguous_negative_map == 0) * (positive_map == 0)

    pos_num = positive_map.flatten(1).sum(dim=1)
    neg_num = ((ambiguous_negative_map == 0) * (positive_map == 0)).flatten(1).sum(dim=1)
    loss = 2 * pos.flatten(1).sum(dim=1) / (pos_num + 1e-7) + neg.flatten(1).sum(dim=1) / (neg_num + 1e-7)
    return loss.mean()
"""
def get_reg_loss(pred, gt, threshold, level=3, window_size=3):
    # Section 3.1: Sigmoid-Based Soft-Margin Loss
    # Replaces the strict (gt > threshold) with the Sigmoid formulation (Equation 2)
    # Assuming tau (scaling factor) is the threshold parameter
    mask = torch.sigmoid((gt / threshold) - 1.0)
    
    loss_ssim = cal_avg_ms_ssim(pred * mask, gt * mask, level=level,
                                window_size=window_size)

    mu_normed = get_normalized_map(pred)
    gt_mu_normed = get_normalized_map(gt)
    
    # Section 3.2: MSE Loss (L2 Regularization)
    # Replaced nn.L1Loss with nn.MSELoss (Equation 4)
    tv_loss = (nn.MSELoss(reduction='none')(mu_normed, gt_mu_normed).sum(1).sum(1).sum(1)).mean(0)

    pred_count = pred.flatten(1).sum(dim=1)
    gt_count = gt.flatten(1).sum(dim=1)
    
    # Also replaced L1 with MSE for the count_loss to be consistent with Section 3.2
    count_loss = nn.MSELoss()(pred_count, gt_count)

    return loss_ssim + 0.1 * tv_loss + 0.01 * count_loss


def RRC_loss(simi, ambiguous_negative_map, positive_map, alpha=1.0):
    pos = (1 - simi) * positive_map.float()
    
    # Section 3.3: Contrastive loss in LRRC (Equation 5)
    # Adding the quadratic penalty term: max(0, simi) + alpha * [max(0, simi)]^2
    clamped_simi = torch.clamp(simi, min=0.0)
    contrastive_penalty = clamped_simi + alpha * (clamped_simi ** 2)
    
    neg = contrastive_penalty * (ambiguous_negative_map == 0).float() * (positive_map == 0).float()

    pos_num = positive_map.float().flatten(1).sum(dim=1)
    neg_num = ((ambiguous_negative_map == 0) * (positive_map == 0)).float().flatten(1).sum(dim=1)
    
    loss = 2 * pos.flatten(1).sum(dim=1) / (pos_num + 1e-7) + neg.flatten(1).sum(dim=1) / (neg_num + 1e-7)
    
    return loss.mean()
"""