"""
PyTorch Dataset for LASS_clap training
"""
import json
import random
import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio.transforms as T


class LASSClapDataset(Dataset):
    """
    Dataset for LASS_clap training
    
    Loads:
    - mixture.wav (input)
    - target.wav (ground truth for loss)
    - reference.wav (for CLAP conditioning)
    - metadata.json (prompt text)
    """
    
    def __init__(
        self,
        data_dir,
        config,
        augment=False,
        cache_in_memory=False
    ):
        """
        Args:
            data_dir: directory containing example_xxxxx folders
            config: DataConfig instance
            augment: whether to apply data augmentation
            cache_in_memory: load all data in RAM (faster but memory intensive)
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        
        # Find all examples
        self.examples = sorted(self.data_dir.glob('example_*'))
        
        if len(self.examples) == 0:
            raise ValueError(f"No examples found in {data_dir}")
        
        print(f"Found {len(self.examples)} examples in {data_dir}")
        
        # Spectrogramme transform
        self.mel_spec = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=config.spec_power
        )
        
        # Data augmentation transforms
        if self.augment:
            self.pitch_shift = T.PitchShift(
                sample_rate=config.sample_rate,
                n_steps=0  # Will be set randomly
            )
            self.time_stretch = T.TimeStretch(
                hop_length=config.hop_length,
                n_freq=config.n_fft // 2 + 1
            )
        
        # Cache
        self.cache = {} if cache_in_memory else None
        if cache_in_memory:
            print("Caching dataset in memory...")
            for idx in range(len(self.examples)):
                self.cache[idx] = self._load_example(idx)
            print("✓ Dataset cached")
    
    def __len__(self):
        return len(self.examples)
    
    def _load_example(self, idx):
        """Load a single example from disk"""
        example_dir = self.examples[idx]
        
        # Load audio files
        mixture, sr = torchaudio.load(example_dir / 'mixture.wav')
        target, sr = torchaudio.load(example_dir / 'target.wav')
        reference, sr = torchaudio.load(example_dir / 'reference.wav')
        
        # Ensure correct sample rate
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            mixture = resampler(mixture)
            target = resampler(target)
            reference = resampler(reference)
        
        # Ensure mono
        if mixture.shape[0] > 1:
            mixture = mixture.mean(dim=0, keepdim=True)
        if target.shape[0] > 1:
            target = target.mean(dim=0, keepdim=True)
        if reference.shape[0] > 1:
            reference = reference.mean(dim=0, keepdim=True)
        
        # Load metadata
        with open(example_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        prompt = metadata['prompt']
        
        return {
            'mixture': mixture.squeeze(0),      # (T,)
            'target': target.squeeze(0),        # (T,)
            'reference': reference.squeeze(0),  # (T,)
            'prompt': prompt,
            'example_id': example_dir.name
        }
    
    def _apply_augmentation(self, audio):
        """Apply random data augmentation"""
        if not self.augment:
            return audio
        
        # Random pitch shift
        if random.random() < 0.5:
            n_steps = random.randint(*self.config.pitch_shift_range)
            if n_steps != 0:
                self.pitch_shift.n_steps = n_steps
                audio = self.pitch_shift(audio.unsqueeze(0)).squeeze(0)
        
        # Random time stretch (plus complexe, skip pour l'instant)
        # if random.random() < 0.3:
        #     rate = random.uniform(*self.config.time_stretch_range)
        #     ...
        
        return audio
    
    def _compute_spectrogram(self, audio):
        """
        Compute mel-spectrogram
        
        Args:
            audio: (T,) tensor
            
        Returns:
            spec: (1, n_mels, T') tensor
        """
        # Mel spectrogram
        spec = self.mel_spec(audio)  # (n_mels, T')
        
        # Log scale
        spec = torch.log(spec + self.config.spec_min)
        
        # Add channel dimension
        spec = spec.unsqueeze(0)  # (1, n_mels, T')
        
        return spec
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
            - mixture_spec: (1, n_mels, T') - input spectrogram
            - target_spec: (1, n_mels, T') - target spectrogram (for loss)
            - reference_audio: (ref_samples,) - for CLAP conditioning
            - prompt: str - text prompt for CLAP
        """
        # Load from cache or disk
        if self.cache_in_memory:
            example = self.cache[idx]
        else:
            example = self._load_example(idx)
        
        mixture = example['mixture']
        target = example['target']
        reference = example['reference']
        prompt = example['prompt']
        
        # Data augmentation (on mixture and target together to keep consistency)
        if self.augment:
            # Apply same augmentation to mixture and target
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            mixture = self._apply_augmentation(mixture)
            
            random.seed(seed)
            target = self._apply_augmentation(target)
            
            # Different augmentation for reference (it's a different excerpt)
            reference = self._apply_augmentation(reference)
        
        # Compute spectrograms
        mixture_spec = self._compute_spectrogram(mixture)
        target_spec = self._compute_spectrogram(target)
        
        # Reference stays as waveform for CLAP
        # CLAP expects raw audio
        
        return {
            'mixture_spec': mixture_spec,      # (1, n_mels, T')
            'target_spec': target_spec,        # (1, n_mels, T')
            'reference_audio': reference,      # (ref_samples,)
            'prompt': prompt,                  # str
            'example_id': example['example_id']
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length and text prompts
    """
    # Find max spectrogram length
    max_spec_len = max(item['mixture_spec'].shape[2] for item in batch)
    
    batch_size = len(batch)
    n_mels = batch[0]['mixture_spec'].shape[1]
    
    # Initialize padded tensors
    mixture_specs = torch.zeros(batch_size, 1, n_mels, max_spec_len)
    target_specs = torch.zeros(batch_size, 1, n_mels, max_spec_len)
    
    # Find max reference audio length
    max_ref_len = max(item['reference_audio'].shape[0] for item in batch)
    reference_audios = torch.zeros(batch_size, max_ref_len)
    
    prompts = []
    example_ids = []
    
    for i, item in enumerate(batch):
        # Pad spectrograms
        spec_len = item['mixture_spec'].shape[2]
        mixture_specs[i, :, :, :spec_len] = item['mixture_spec']
        target_specs[i, :, :, :spec_len] = item['target_spec']
        
        # Pad reference audio
        ref_len = item['reference_audio'].shape[0]
        reference_audios[i, :ref_len] = item['reference_audio']
        
        prompts.append(item['prompt'])
        example_ids.append(item['example_id'])
    
    return {
        'mixture_spec': mixture_specs,      # (B, 1, n_mels, T')
        'target_spec': target_specs,        # (B, 1, n_mels, T')
        'reference_audio': reference_audios, # (B, max_ref_len)
        'prompts': prompts,                 # List[str]
        'example_ids': example_ids          # List[str]
    }


# Test the dataset
if __name__ == "__main__":
    from config import DataConfig
    
    config = DataConfig()
    
    # Test dataset
    dataset = LASSClapDataset(
        data_dir="data/processed/train_midi",
        config=config,
        augment=True,
        cache_in_memory=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  mixture_spec: {sample['mixture_spec'].shape}")
    print(f"  target_spec: {sample['target_spec'].shape}")
    print(f"  reference_audio: {sample['reference_audio'].shape}")
    print(f"  prompt: {sample['prompt']}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  mixture_spec: {batch['mixture_spec'].shape}")
    print(f"  target_spec: {batch['target_spec'].shape}")
    print(f"  reference_audio: {batch['reference_audio'].shape}")
    print(f"  prompts: {batch['prompts']}")
    
    print("\n✓ Dataset test passed!")
    