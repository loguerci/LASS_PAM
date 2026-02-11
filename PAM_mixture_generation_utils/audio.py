"""
Audio loading and mixing utilities
Based on LASS mixing approach
"""
import numpy as np
import librosa
import soundfile as sf
import random
from pathlib import Path
from IPython.display import Audio, display

SAMPLE_RATE = 16000
MIX_DURATION = 10.0

# Audio quality thresholds
MIN_RMS = 1e-4
MIN_ACTIVITY_RATIO = 0.8
ACTIVITY_THRESHOLD = 0.001



def get_active_audio(audio : np.ndarray, ACTIVITY_THRESHOLD = ACTIVITY_THRESHOLD) -> (np.ndarray | None):
    """Return active part of audio based on simple thresholding"""
    if audio is None or len(audio) == 0:
        print("audio.get_active_audio : input audio is None or empty, returning None")
        return None
    active_samples = np.convolve([1, 1], np.abs(audio), mode='same') > ACTIVITY_THRESHOLD

    if np.all(~(active_samples.copy())):
        print("audio.get_active_audio : no active samples found, returning None")
        return None
    return audio[active_samples]


def load_audio_segment(path, sr=SAMPLE_RATE) -> (np.ndarray | None):
    """Load audio segment and return active part"""
    try:
        audio, _ = librosa.load(path, sr=sr)
        if audio is None:
            print(f"audio.load_audio_segment : failed to load audio from {path} (librosa returned None)")
            return None
        active_audio = get_active_audio(audio)
        if len(active_audio) == 0:
            print(f"audio.load_audio_segment : no active audio found in {path}")
            return None
        return np.nan_to_num(active_audio, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"audio.load_audio_segment : error loading {path}: {e}")
        return None

def scatter_audio_segments(segments :list[np.ndarray], no_process_segments : list[np.ndarray] = [], mix_duration_s : float = MIX_DURATION, sr :int = SAMPLE_RATE, mix_division=10, instance_probability = .5, max_seg_duration_s = 5.0) -> np.ndarray:
    length = int(mix_duration_s * sr)
    max_seg_length = int(max_seg_duration_s * sr)
    mix = np.zeros(length)
    mix_segments = [(i * length//mix_division, (i+1) * length//mix_division) for i in range(mix_division)]
    for seg in segments:
        if seg is None:
            print("audio.scatter_audio_segments : one of the segments is None, skipping this segment")
            continue
        if len(seg) > max_seg_length:
            seg = seg[:min(len(seg), max_seg_length)]
        miniseg_length = min(len(seg), max_seg_length)
        if miniseg_length ==  0:
            print("audio.scatter_audio_segments : one of the segments is empty after trimming, skipping this segment")
            continue
        scatter_indices = []
        for i, j in mix_segments:
            # Fix: ensure k doesn't go beyond the valid range
            k = random.randint(0, max(0, len(seg) - miniseg_length))
            miniseg = seg[k:k + miniseg_length]
            instance_success = random.random() < instance_probability
            if instance_success and (scatter_indices and j > scatter_indices[-1] + miniseg_length) or not scatter_indices:
                if not scatter_indices:
                    scatter_indices.append(random.randint(i, j))
                else :
                    scatter_indices.append(random.randint(max(i, scatter_indices[-1] + miniseg_length), j))
        
        if not scatter_indices:
            scatter_indices.append(random.randint(0, max(0, length - miniseg_length)))
        
        for idx in scatter_indices:
            remaining = length - idx
            if remaining <= 0:
                continue
            # Fix: use actual length of miniseg, not miniseg_length
            seg_len = min(len(miniseg), remaining)
            if seg_len <= 0:
                continue
            mix[idx:idx + seg_len] += np.nan_to_num(
                    miniseg[:seg_len],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )    
    for seg in no_process_segments:
        mix += seg[:length]
        
    mix = .9 * mix / (np.max(np.abs(mix)) + 1e-6)
    return mix

def save_audio(path, audio, sr=SAMPLE_RATE):
    try:
        sf.write(path, audio, sr)
    except Exception as e:
        print(f"audio.save_audio : error saving {path}: {e}")


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

#def load_audio_segment(path, duration=MIX_DURATION, sr=SAMPLE_RATE, max_attempts=10):
#    """Load valid audio segment with retry"""
#    info = sf.info(path)
#    total = info.duration
#    print("audio.load_audio_segment : loading segment from {path} with total duration {total:.2f}s")
#
#    if total <= duration:
#        audio, _ = librosa.load(path, sr=sr)
#        if audio is None:
#            print(f"audio.load_audio_segment : failed to load audio from {path} (librosa returned None)")
#            return None
#        if not is_audio_valid(audio):
#            print(f"audio.load_audio_segment : audio from {path} is too short and invalid")
#            return None
#        pad = int(duration * sr) - len(audio)
#        if pad > 0:
#            audio = np.pad(audio, (0, pad))
#        return audio
#
#    for attempt in range(max_attempts):
#        offset = random.uniform(0, total - duration)
#        audio, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
#        
#        if is_audio_valid(audio):
#            return audio
#    
#    print(f"audio.load_audio_segment : failed to load valid segment from {path} after {max_attempts} attempts")
#    return None


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
    if audio is None or len(audio) == 0:
        print("audio.normalize_energy : input audio is None or empty, returning None")
        return None
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