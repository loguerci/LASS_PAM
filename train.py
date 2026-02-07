"""
Training script for LASS_clap
Adapted for UNetRes_FiLM architecture
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
from utils.losses import get_loss_function, compute_metrics


class LASSClapTrainer:
    """Trainer class for LASS_clap"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup directories
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.training.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=self.log_dir / f"{config.exp_name}_{self.timestamp}"
        )
        
        # Model
        print("\nInitializing model...")
        self.model = LASS_clap(device=self.device)
        self.model = self.model.to(self.device)
        
        # Load pretrained weights if specified
        if config.model.load_pretrained and Path(config.model.pretrained_lass_ckpt).exists():
            print(f"Loading pretrained LASS weights from {config.model.pretrained_lass_ckpt}")
            self.model.load_pretrained_unet(config.model.pretrained_lass_ckpt, strict=False)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Loss function
        self.criterion = get_loss_function(config.training.loss_type)
        
        # Optimizer (will be recreated in _apply_finetune_schedule)
        self.optimizer = None
        self._initialize_optimizer()
        
        # Learning rate scheduler
        self.scheduler = None
        if config.training.use_scheduler:
            self._initialize_scheduler()
        
        # Warmup scheduler
        self.warmup_scheduler = None
        if config.training.warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=config.training.warmup_epochs
            )
        
        # Datasets
        print("\nLoading datasets...")
        self.train_dataset = LASSClapDataset(
            data_dir=config.data.train_dir,
            config=config.data,
            augment=config.data.use_augmentation,
            cache_in_memory=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Validation dataset (if exists)
        self.val_loader = None
        if Path(config.data.val_dir).exists():
            self.val_dataset = LASSClapDataset(
                data_dir=config.data.val_dir,
                config=config.data,
                augment=False,
                cache_in_memory=False
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.training.num_workers,
                collate_fn=collate_fn,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            print(f"✓ Validation dataset: {len(self.val_dataset)} examples")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Save config
        self._save_config()
        
        # Print summary
        self._print_summary()
    
    def _initialize_optimizer(self):
        """Initialize optimizer with trainable parameters"""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=self.config.training.betas
        )
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.config.training.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
    
    def _save_config(self):
        """Save training configuration"""
        config_dict = {
            'exp_name': self.config.exp_name,
            'timestamp': self.timestamp,
            'device': str(self.device),
            'num_params': self.num_params,
            'trainable_params': self.trainable_params,
            'data': {
                'train_dir': self.config.data.train_dir,
                'sample_rate': self.config.data.sample_rate,
                'n_fft': self.config.data.n_fft,
                'hop_length': self.config.data.hop_length,
                'n_mels': self.config.data.n_mels,
                'duration': self.config.data.duration,
            },
            'model': {
                'clap_ckpt': self.config.model.clap_ckpt,
                'clap_freeze': self.config.model.clap_freeze,
                'load_pretrained': self.config.model.load_pretrained,
            },
            'training': {
                'batch_size': self.config.training.batch_size,
                'num_epochs': self.config.training.num_epochs,
                'learning_rate': self.config.training.learning_rate,
                'loss_type': self.config.training.loss_type,
                'scheduler_type': self.config.training.scheduler_type,
                'finetune_schedule': self.config.training.finetune_schedule,
            }
        }
        
        config_path = self.checkpoint_dir / f"config_{self.timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Config saved: {config_path}")
    
    def _print_summary(self):
        """Print training summary"""
        print(f"\n{'='*60}")
        print(f"LASS_clap Training Configuration")
        print(f"{'='*60}")
        print(f"Experiment: {self.config.exp_name}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {self.num_params:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")
        print(f"\nDataset:")
        print(f"  Train: {len(self.train_dataset)} examples")
        if self.val_loader:
            print(f"  Val: {len(self.val_dataset)} examples")
        print(f"\nTraining:")
        print(f"  Epochs: {self.config.training.num_epochs}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  Learning rate: {self.config.training.learning_rate}")
        print(f"  Loss: {self.config.training.loss_type}")
        print(f"  Scheduler: {self.config.training.scheduler_type}")
        print(f"{'='*60}\n")
    
    def _get_encoder_blocks(self):
        """Get list of encoder blocks from UNetRes_FiLM"""
        return [
            self.model.UNet.encoder_block1,
            self.model.UNet.encoder_block2,
            self.model.UNet.encoder_block3,
            self.model.UNet.encoder_block4,
            self.model.UNet.encoder_block5,
            self.model.UNet.encoder_block6,
        ]
    
    def _get_decoder_blocks(self):
        """Get list of decoder blocks from UNetRes_FiLM"""
        return [
            self.model.UNet.decoder_block1,
            self.model.UNet.decoder_block2,
            self.model.UNet.decoder_block3,
            self.model.UNet.decoder_block4,
            self.model.UNet.decoder_block5,
            self.model.UNet.decoder_block6,
        ]
    
    def _apply_finetune_schedule(self, epoch):
        """       
        Phase 1 (epochs 0-15): 
          - Freeze ALL encoder blocks
          - Freeze early decoder blocks (0, 1, 2)
          - Train only: later decoder blocks + after_conv layers
        
        Phase 2 (epochs 16-30):
          - Freeze ALL encoder blocks (still)
          - Freeze only decoder_block1 (index 0)
          - Unfreeze decoder_blocks 2-6
        
        Phase 3 (epochs 31+):
          - Freeze ALL encoder blocks (TOUJOURS!)
          - Unfreeze ALL decoder blocks
          - Full decoder fine-tuning
        """
        schedule = self.config.training.finetune_schedule
        
        # Find current phase
        current_phase = None
        for phase_name, phase_config in schedule.items():
            start_epoch, end_epoch = phase_config['epochs']
            if start_epoch <= epoch < end_epoch:
                current_phase = phase_name
                break
        
        if current_phase is None:
            return  # No change needed
        
        phase_config = schedule[current_phase]

        encoder_blocks = self._get_encoder_blocks()
        for block in encoder_blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        for param in self.model.UNet.conv_block7.parameters():
            param.requires_grad = False

        decoder_blocks = self._get_decoder_blocks()
        freeze_decoder_indices = phase_config.get('freeze_decoder_blocks', [])
        
        for i, block in enumerate(decoder_blocks):
            freeze = i in freeze_decoder_indices
            for param in block.parameters():
                param.requires_grad = not freeze
        
        for param in self.model.UNet.after_conv_block1.parameters():
            param.requires_grad = True
        for param in self.model.UNet.after_conv2.parameters():
            param.requires_grad = True

        for param in self.model.clap_conditioner.clap.parameters():
            param.requires_grad = False
        
        # Projection layer CLAP est trainable
        for param in self.model.proj.parameters():
            param.requires_grad = True

        phase_lr = phase_config.get('lr', self.config.training.learning_rate)
        
        # Recreate optimizer with new trainable parameters
        self._initialize_optimizer()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = phase_lr
        
        # Recreate scheduler if needed
        if self.config.training.use_scheduler:
            self._initialize_scheduler()
  
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"Fine-tuning Phase: {current_phase} (Epoch {epoch+1})")
        print(f"{'='*60}")
        print(f"Encoder blocks: FROZEN (all 6 blocks)")
        print(f"Bottleneck (conv_block7): FROZEN")
        print(f"Decoder blocks frozen: {freeze_decoder_indices}")
        print(f"Decoder blocks trainable: {[i for i in range(6) if i not in freeze_decoder_indices]}")
        print(f"After conv layers: TRAINABLE")
        print(f"CLAP: FROZEN")
        print(f"CLAP projection: TRAINABLE")
        print(f"\nLearning rate: {phase_lr:.6f}")
        print(f"Trainable params: {trainable:,} / {self.num_params:,} ({100*trainable/self.num_params:.1f}%)")
        print(f"{'='*60}\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Apply fine-tuning schedule at phase transitions
        if epoch in [0, 16, 31]:
            self._apply_finetune_schedule(epoch)
        
        epoch_metrics = {
            'loss': 0.0,
            'batch_count': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            mixture_spec = batch['mixture_spec'].to(self.device)  # (B, 1, F, T)
            target_spec = batch['target_spec'].to(self.device)    # (B, 1, F, T)
            reference_audio = batch['reference_audio'].to(self.device)  # (B, ref_len)
            prompts = batch['prompts']  # List[str]
            
            # Forward pass
            # LASS_clap.forward(x, ref, caption) -> UNet.forward(x, cond_vec, dec_cond_vec)
            pred_mask = self.model(mixture_spec, reference_audio, prompts)  # (B, 1, F, T)
            
            # Apply mask to mixture to get separated spectrogram
            pred_spec = pred_mask * mixture_spec
            
            # Compute loss (compare with target_spec)
            loss = self.criterion(pred_spec, target_spec)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = mixture_spec.size(0)
            epoch_metrics['loss'] += loss.item() * batch_size
            epoch_metrics['batch_count'] += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.training.log_every == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Compute epoch averages
        avg_loss = epoch_metrics['loss'] / epoch_metrics['batch_count']
        
        # Log epoch metrics
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'sdr': 0.0,
            'batch_count': 0
        }
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            mixture_spec = batch['mixture_spec'].to(self.device)
            target_spec = batch['target_spec'].to(self.device)
            reference_audio = batch['reference_audio'].to(self.device)
            prompts = batch['prompts']
            
            # Forward pass
            pred_mask = self.model(mixture_spec, reference_audio, prompts)
            pred_spec = pred_mask * mixture_spec
            
            # Compute loss
            loss = self.criterion(pred_spec, target_spec)
            
            # Compute metrics
            metrics = compute_metrics(pred_spec, target_spec)
            
            batch_size = mixture_spec.size(0)
            val_metrics['loss'] += loss.item() * batch_size
            val_metrics['sdr'] += metrics['sdr'] * batch_size
            val_metrics['batch_count'] += batch_size
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute averages
        avg_loss = val_metrics['loss'] / val_metrics['batch_count']
        avg_sdr = val_metrics['sdr'] / val_metrics['batch_count']
        
        # Log
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/sdr', avg_sdr, epoch)
        
        return avg_loss, avg_sdr
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config.training.save_every == 0:
            checkpoint_path = self.checkpoint_dir / f"lass_clap_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Keep only last N checkpoints
            self._cleanup_old_checkpoints()
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "lass_clap_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best checkpoint saved: {best_path}")
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("lass_clap_epoch_*.pth"))
        
        if len(checkpoints) > self.config.training.keep_last_n:
            for ckpt in checkpoints[:-self.config.training.keep_last_n]:
                ckpt.unlink()
                print(f"  Removed old checkpoint: {ckpt.name}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ Resumed from epoch {self.current_epoch + 1}")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_results = None
            if (epoch + 1) % self.config.training.val_every == 0:
                val_results = self.validate(epoch)
            
            # Learning rate scheduling
            if epoch < self.config.training.warmup_epochs and self.warmup_scheduler:
                self.warmup_scheduler.step()
            elif self.scheduler:
                if self.config.training.scheduler_type == "plateau" and val_results:
                    self.scheduler.step(val_results[0])
                elif self.config.training.scheduler_type == "cosine":
                    self.scheduler.step()
            
            # Log epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            log_str = f"Epoch {epoch+1}/{self.config.training.num_epochs} | Train Loss: {train_loss:.4f}"
            
            if val_results:
                val_loss, val_sdr = val_results
                log_str += f" | Val Loss: {val_loss:.4f} | Val SDR: {val_sdr:.2f} dB"
                
                # Check if best
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    log_str += " ✓ (best)"
            else:
                is_best = False
            
            log_str += f" | LR: {current_lr:.6f}"
            print(log_str)
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
        
        # Save final model
        final_path = self.checkpoint_dir / "lass_clap_final.pth"
        torch.save({
            'epoch': self.config.training.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, final_path)
        
        self.writer.close()
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Final model saved: {final_path}")
        if self.val_loader:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train LASS_clap model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='Training data directory')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override with command line arguments
    if args.train_dir:
        config.data.train_dir = args.train_dir
    if args.val_dir:
        config.data.val_dir = args.val_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.exp_name:
        config.exp_name = args.exp_name
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Create trainer
    trainer = LASSClapTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()