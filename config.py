"""
Configuration for LASS_clap training
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    train_dir: str = "data/processed/train"
    val_dir: str = "data/processed/val" 
    
    # Audio parameters
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    duration: float = 10.0  # seconds
    
    # CLAP reference
    ref_duration: float = 10.0  # seconds for reference audio
    
    # Spectrogramme
    spec_min: float = 1e-8  # Pour éviter log(0)
    spec_power: float = 1.0  # 1.0 = magnitude, 2.0 = power
    
    # Data augmentation
    use_augmentation: bool = True
    pitch_shift_range: tuple = (-2, 2)  # semitones
    time_stretch_range: tuple = (0.9, 1.1)


@dataclass
class ModelConfig:
    """Model configuration"""
    # Architecture
    channels: int = 1
    cond_embedding_dim: int = 256
    
    # CLAP
    clap_ckpt: str = "pretrained/music_audioset_epoch_15_esc_90.14.pt"
    clap_freeze: bool = True
    use_clap_audio: bool = True
    use_clap_text: bool = True
    
    # Pretrained weights
    pretrained_lass_ckpt: str = "pretrained/LASSNet.pt"
    load_pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic
    batch_size: int = 16
    num_epochs: int = 50
    num_workers: int = 4
    
    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss
    loss_type: str = "l1"  # "l1", "mse", "spectral"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/lass_clap"
    save_every: int = 5  # epochs
    keep_last_n: int = 3  # number of checkpoints to keep
    
    # Logging
    log_dir: str = "logs/lass_clap"
    log_every: int = 10  # steps
    
    # Validation
    val_every: int = 1  # epochs
    


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Device
    device: str = "cpu"  # "cuda" or "cpu"
    seed: int = 42
    
    # Experiment name
    exp_name: str = "lass_clap_baseline"