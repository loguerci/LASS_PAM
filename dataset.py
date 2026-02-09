"""
Dataset for LASS_clap training
"""
import json
import random
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from utils.stft import STFT


class LASSClapDataset(Dataset):
    """
    Dataset for LASS_clap training
    """
    
    def __init__(
        self,
        data_dir,
        sample_rate=16000,
        segment_samples=None,  # None = use full duration
        augment=False,
        cache_in_memory=False
    ):
        """
        Args:
            data_dir: directory containing example_xxxxx folders
            sample_rate: target sample rate (should match LASS config)
            segment_samples: length of audio segment in samples (None = full)
            augment: whether to apply data augmentation
            cache_in_memory: load all data in RAM (faster but memory intensive)
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.augment = augment
        self.cache_in_memory = cache_in_memory
        
        # Find all examples
        self.examples = sorted(self.data_dir.glob('example_*'))
        
        if len(self.examples) == 0:
            raise ValueError(f"No examples found in {data_dir}")
        
        print(f"Found {len(self.examples)} examples in {data_dir}")
        
        # Cache
        self.cache = {} if cache_in_memory else None
        if cache_in_memory:
            print("Caching dataset in memory...")
            for idx in range(len(self.examples)):
                self.cache[idx] = self._load_example(idx)
            print("✓ Dataset cached")
    
    def __len__(self):
        return len(self.examples)
    
    def _load_audio(self, audio_path, target_length=None):
        """
        Load audio file and ensure correct format
        
        Args:
            audio_path: path to .wav file
            target_length: desired length in samples (None = keep original)
        
        Returns:
            audio: (num_samples,) tensor
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)
        
        # Adjust length if specified
        if target_length is not None:
            current_length = audio.shape[0]
            
            if current_length > target_length:
                # Random crop (for training) or center crop
                if self.augment:
                    start = random.randint(0, current_length - target_length)
                else:
                    start = (current_length - target_length) // 2
                audio = audio[start:start + target_length]
            
            elif current_length < target_length:
                # Pad with zeros
                pad_length = target_length - current_length
                audio = torch.nn.functional.pad(audio, (0, pad_length))
        
        return audio
    
    def _load_example(self, idx):
        """Load a single example from disk"""
        example_dir = self.examples[idx]
        
        # Load metadata
        metadata_file = example_dir / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        prompt = metadata['prompt']
        
        # Determine target length (if segment_samples is set)
        target_length = self.segment_samples
        
        # Load audio files
        mixture = self._load_audio(
            example_dir / 'mixture.wav',
            target_length=target_length
        )
        
        target = self._load_audio(
            example_dir / 'target.wav',
            target_length=target_length
        )
        
        reference = self._load_audio(
            example_dir / 'reference.wav',
            target_length=target_length
        )
        
        return {
            'mixture': mixture,      # (num_samples,)
            'target': target,        # (num_samples,)
            'reference': reference,  # (num_samples,)
            'prompt': prompt,
            'example_id': example_dir.name,
            'metadata': metadata
        }
    
    def _apply_augmentation(self, audio):
        """
        Apply random data augmentation
        
        TODO: Add more augmentations as needed:
        - Pitch shift
        - Time stretch
        - Add noise
        - Volume change
        """
        if not self.augment:
            return audio
        
        # Random volume scaling
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            audio = audio * scale
        
        # TODO: Add more augmentations here
        
        return audio
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
            - mixture: (num_samples,) - mixed audio waveform
            - target: (num_samples,) - target source waveform (for loss)
            - reference: (num_samples,) - reference audio for CLAP
            - prompt: str - text prompt for CLAP
            - example_id: str - identifier
        """
        # Load from cache or disk
        if self.cache_in_memory:
            example = self.cache[idx].copy()  # Copy to avoid modifying cache
        else:
            example = self._load_example(idx)
        
        mixture = example['mixture']
        target = example['target']
        reference = example['reference']
        prompt = example['prompt']
        
        # Data augmentation
        # Apply SAME augmentation to mixture and target (to maintain consistency)
        if self.augment:
            seed = random.randint(0, 2**32 - 1)
            
            # Augment mixture and target with same seed
            random.seed(seed)
            mixture = self._apply_augmentation(mixture)
            
            random.seed(seed)
            target = self._apply_augmentation(target)
            
            # Different augmentation for reference (it's a different excerpt)
            reference = self._apply_augmentation(reference)
        
        # Return raw waveforms (STFT will be done in training loop)
        return {
            'mixture': mixture,          # (num_samples,) - for STFT in training
            'target': target,            # (num_samples,) - for STFT in training
            'reference': reference,      # (num_samples,) - for CLAP
            'prompt': prompt,            # str
            'example_id': example['example_id']
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length audio and text prompts
    
    Args:
        batch: list of dicts from __getitem__
    
    Returns:
        dict with batched tensors
    """
    # Find max length
    max_mixture_len = max(item['mixture'].shape[0] for item in batch)
    max_target_len = max(item['target'].shape[0] for item in batch)
    max_ref_len = max(item['reference'].shape[0] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    mixtures = torch.zeros(batch_size, max_mixture_len)
    targets = torch.zeros(batch_size, max_target_len)
    references = torch.zeros(batch_size, max_ref_len)
    
    prompts = []
    example_ids = []
    
    for i, item in enumerate(batch):
        # Pad audio
        mixture_len = item['mixture'].shape[0]
        mixtures[i, :mixture_len] = item['mixture']
        
        target_len = item['target'].shape[0]
        targets[i, :target_len] = item['target']
        
        ref_len = item['reference'].shape[0]
        references[i, :ref_len] = item['reference']
        
        prompts.append(item['prompt'])
        example_ids.append(item['example_id'])
    
    return {
        'mixture': mixtures,        # (B, num_samples)
        'target': targets,          # (B, num_samples)
        'reference': references,    # (B, num_samples)
        'prompts': prompts,         # List[str]
        'example_ids': example_ids  # List[str]
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("Testing LASSClapDataset...")
    print("="*60)
    
    # Create dataset
    dataset = LASSClapDataset(
        data_dir="data/processed/train",
        sample_rate=16000,
        segment_samples=16000 * 10,  # 10 seconds
        augment=True,
        cache_in_memory=False
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test a single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    
    print(f"  mixture: {sample['mixture'].shape}")
    print(f"  target: {sample['target'].shape}")
    print(f"  reference: {sample['reference'].shape}")
    print(f"  prompt: {sample['prompt']}")
    print(f"  example_id: {sample['example_id']}")
    
    # Test dataloader
    print("\nTesting dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for training
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    
    print(f"\nBatch shapes:")

    mixture = batch['mixture'] # (B, 1, F, T)
    target = batch['target']   # (B, 1, F, T)
    reference = batch['reference']  # (B, ref_len)
    prompts = batch['prompts']  # List[str]

    stft = STFT()

    mix_mag, _ = stft.transform(mixture)
    target_mag, _ = stft.transform(target)

    print(f"{mix_mag.shape}")
    print(f"{target_mag.shape}")
    print(f"{batch['prompts'][0]}")
    print(f"{reference.shape}")