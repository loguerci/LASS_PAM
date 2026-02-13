"""
Simple training script for LASS_clap
Train decoder only, freeze encoder + CLAP
"""
from pathlib import Path
from datetime import datetime
import json
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import LASSClapDataset, collate_fn
from model.LASS_clap import LASS_clap
from utils.stft import STFT

CUDA_VISIBLE_DEVICES=1


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda'
        
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{config.training.log_dir}/{config.exp_name}_{self.timestamp}")
        
        # Model
        self.model = LASS_clap(device=self.device).to(self.device)
        
        # Load pretrained
        if config.model.load_pretrained:
            ckpt_path = Path(config.model.pretrained_lass_ckpt)
            if ckpt_path.exists():
                print(f"Loading checkpoint: {ckpt_path}")
                self.model.load_pretrained_unet(str(ckpt_path), strict=False)
        
        # Freeze encoder + CLAP, train decoder only
        self._setup_trainable_params()
        
        # Loss
        self.criterion = torch.nn.L1Loss()
        
        # Optimizer
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.AdamW(
            trainable,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs
        )
        
        # STFT
        self.stft = STFT().to(self.device)
        
        # Data
        self.train_loader = self._get_dataloader(config.data.train_dir, shuffle=True)
        self.val_loader = self._get_dataloader(config.data.val_dir, shuffle=False) if Path(config.data.val_dir).exists() else None
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        print(f"\nTrainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Dataset: {len(self.train_loader.dataset)} train examples\n")
    
    def _setup_trainable_params(self):
        """Freeze encoder + CLAP, train decoder only"""
        # Freeze ALL encoder blocks
        for block in [
            self.model.UNet.encoder_block1,
            self.model.UNet.encoder_block2,
            self.model.UNet.encoder_block3,
            self.model.UNet.encoder_block4,
            self.model.UNet.encoder_block5,
            self.model.UNet.encoder_block6,
            self.model.UNet.conv_block7  # bottleneck
        ]:
            for param in block.parameters():
                param.requires_grad = False
        
        # Train ALL decoder blocks
        for block in [
            self.model.UNet.decoder_block1,
            self.model.UNet.decoder_block2,
            self.model.UNet.decoder_block3,
            self.model.UNet.decoder_block4,
            self.model.UNet.decoder_block5,
            self.model.UNet.decoder_block6,
            self.model.UNet.after_conv_block1,
            self.model.UNet.after_conv2
        ]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Freeze CLAP
        for param in self.model.clap_conditioner.clap.parameters():
            param.requires_grad = False
        
        # Train CLAP projection
        for param in self.model.proj.parameters():
            param.requires_grad = True
    
    def _get_dataloader(self, data_dir, shuffle):
        dataset = LASSClapDataset(
            data_dir=data_dir,
            sample_rate=self.config.data.sample_rate,
            segment_samples=int(self.config.data.duration * self.config.data.sample_rate),
            augment=shuffle,
            cache_in_memory=False
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            mixture = batch['mixture'].to(self.device)
            target = batch['target'].to(self.device)
            reference = batch['reference'].to(self.device)
            prompts = batch['prompts']
            
            # STFT
            mix_mag, _ = self.stft.transform(mixture)
            target_mag, _ = self.stft.transform(target)
            
            # Forward
            mask = self.model(mix_mag, reference, prompts)
            pred = mask * mix_mag
            
            # Loss
            loss = self.criterion(pred, target_mag)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.global_step += 1
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            mixture = batch['mixture'].to(self.device)
            target = batch['target'].to(self.device)
            reference = batch['reference'].to(self.device)
            prompts = batch['prompts']
            
            mix_mag, _ = self.stft.transform(mixture)
            target_mag, _ = self.stft.transform(target)
            
            mask = self.model(mix_mag, reference, prompts)
            pred = mask * mix_mag
            
            loss = self.criterion(pred, target_mag)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, is_best=False):
        ckpt_dir = Path(self.config.training.checkpoint_dir)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if (epoch + 1) % self.config.training.save_every == 0:
            path = ckpt_dir / f"ckpt_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
        
        if is_best:
            torch.save(checkpoint, ckpt_dir / "best.pth")
    
    def train(self):
        print("Starting training...\n")
        
        for epoch in range(self.config.training.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            
            log = f"Epoch {epoch+1}/{self.config.training.num_epochs} | Train: {train_loss:.4f}"
            
            is_best = False
            if val_loss is not None:
                log += f" | Val: {val_loss:.4f}"
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = True
                    log += " ✓"
                self.writer.add_scalar('val/loss', val_loss, epoch)
            
            print(log)
            self.save_checkpoint(epoch, is_best)
        
        torch.save(self.model.state_dict(), Path(self.config.training.checkpoint_dir) / "final.pth")
        self.writer.close()
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default=None)
    parser.add_argument('--val-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    config = Config()
    if args.train_dir: config.data.train_dir = args.train_dir
    if args.val_dir: config.data.val_dir = args.val_dir
    if args.epochs: config.training.num_epochs = args.epochs
    if args.batch_size: config.training.batch_size = args.batch_size
    if args.lr: config.training.learning_rate = args.lr
    config.device = args.device
    
    torch.manual_seed(42)
    Trainer(config).train()


if __name__ == "__main__":
    main()