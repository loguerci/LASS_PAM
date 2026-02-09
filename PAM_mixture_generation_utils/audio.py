"""
Audio loading and mixing utilities
Based on LASS mixing approach
"""
import numpy as np
import librosa
import soundfile as sf
import random
from pathlib import Path

SAMPLE_RATE = 16000
MIX_DURATION = 10.0

# Audio quality thresholds
MIN_RMS = 1e-4
MIN_ACTIVITY_RATIO = 0.8
ACTIVITY_THRESHOLD = 0.01

# =============================================================================
# AUDIO VALIDATION
# =============================================================================

def is_audio_valid(audio, min_rms=MIN_RMS, min_activity_ratio=MIN_ACTIVITY_RATIO):
    """Check if audio segment has sufficient content"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < min_rms:
        return False
    
    active_samples = np.abs(audio) > ACTIVITY_THRESHOLD
    activity_ratio = np.sum(active_samples) / len(audio)
    
    return activity_ratio >= min_activity_ratio


# =============================================================================
# AUDIO LOADING
# =============================================================================

def load_audio_segment(path, duration=MIX_DURATION, sr=SAMPLE_RATE, max_attempts=10):
    """Load valid audio segment with retry"""
    info = sf.info(path)
    total = info.duration

    if total <= duration:
        audio, _ = librosa.load(path, sr=sr)
        if not is_audio_valid(audio):
            print(f"audio.load_audio_segment : audio from {path} is too short and invalid")
            return None
        pad = int(duration * sr) - len(audio)
        if pad > 0:
            audio = np.pad(audio, (0, pad))
        return audio

    for attempt in range(max_attempts):
        offset = random.uniform(0, total - duration)
        audio, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
        
        if is_audio_valid(audio):
            return audio
    
    print(f"audio.load_audio_segment : failed to load valid segment from {path} after {max_attempts} attempts")
    return None


def load_two_different_segments(path, duration=MIX_DURATION, sr=SAMPLE_RATE, max_attempts=10):
    """Load two different valid segments from same file"""
    info = sf.info(path)
    total = info.duration
    
    min_required = duration * 2 + 5.0
    
    if total < min_required:
        audio = load_audio_segment(path, duration, sr, max_attempts)
        if audio is None:
            print(f"audio.load_two_different_segments : failed to load valid segment from {path}")
            return None, None
        return audio, audio.copy()
    
    for attempt in range(max_attempts):
        offset1 = random.uniform(0, total - duration * 2 - 5)
        audio1, _ = librosa.load(path, sr=sr, offset=offset1, duration=duration)
        
        if not is_audio_valid(audio1):
            print(f"audio.load_two_different_segments : attempt {attempt+1}: invalid first segment from {path}")
            continue
        
        offset2 = offset1 + duration + random.uniform(5, total - offset1 - duration * 2)
        if offset2 + duration > total:
            print(f"audio.load_two_different_segments : attempt {attempt+1}: invalid second segment offset for {path}")
            continue
            
        audio2, _ = librosa.load(path, sr=sr, offset=offset2, duration=duration)
        
        if is_audio_valid(audio2):
            return audio1, audio2
    print(f"audio.load_two_different_segments : failed to load two valid segments from {path} after {max_attempts} attempts")
    return None, None


# =============================================================================
# MIXING (LASS approach)
# =============================================================================

def normalize_energy(audio, alpha=1.0):
    """Normalize audio to [-alpha, alpha] range"""
    val_max = np.max(np.abs(audio))
    if val_max < 1e-8:
        return audio
    return (audio / val_max) * alpha


def unify_energy(*audios):
    """Normalize multiple audios to same energy level"""
    max_amp = max(np.max(np.abs(audio)) for audio in audios)
    if max_amp < 1e-8:
        return audios
    mix_scale = 1.0 / max_amp
    return [audio * mix_scale for audio in audios]


def mix_with_snr(target, background, snr_db_low=-2, snr_db_high=8):
    """
    Mix target and background with random SNR
    Based on LASS add_noise_and_scale approach
    
    Returns:
        mixture, target_scaled, background_scaled or (None, None, None)
    """
    # Validate inputs
    if not is_audio_valid(target) or not is_audio_valid(background):
        print("audio.mix_with_snr : invalid target or background audio for mixing")
        return None, None, None
    
    # Normalize both to [-1, 1]
    target = normalize_energy(target, alpha=1.0)
    background = normalize_energy(background, alpha=1.0)
    
    # Random SNR
    snr_db = random.uniform(snr_db_low, snr_db_high)
    clean_weight = 10 ** (snr_db / 20)
    
    # Apply SNR (reduce background)
    background_scaled = background / clean_weight
    
    # Mix
    mixture = target + background_scaled
    
    # Unify energy to [-1, 1]
    mixture, background_scaled, target = unify_energy(mixture, background_scaled, target)
    
    # Random scale [0.3, 0.9]
    scale = random.uniform(0.3, 0.9)
    mixture = mixture * scale
    target = target * scale
    background_scaled = background_scaled * scale
    
    # Final validation
    if not is_audio_valid(mixture) or not is_audio_valid(target):
        print("audio.mix_with_snr : resulting mixture or target is invalid after mixing")
        return None, None, None
    
    return mixture, target, background_scaled