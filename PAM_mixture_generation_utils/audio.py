# =============================================================================
# AUDIO UTILS
# =============================================================================


import librosa
import numpy as np
import soundfile as sf
import random



SAMPLE_RATE = 16000
MIX_DURATION = 10.0



def load_audio_segment(path, duration, sr=SAMPLE_RATE):
    info = sf.info(path)
    total = info.duration

    if total <= duration:
        audio, _ = librosa.load(path, sr=sr)
        pad = int(duration * sr) - len(audio)
        if pad > 0:
            audio = np.pad(audio, (0, pad))
        return audio

    offset = random.uniform(0, total - duration)
    audio, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return audio


def normalize(audio, target_db=-20.0):
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    target_rms = 10 ** (target_db / 20)
    return audio * (target_rms / rms)


def mix_with_snr(target, background, snr_db):
    target = normalize(target)
    background = normalize(background)

    snr = 10 ** (snr_db / 20)
    tp = np.mean(target**2)
    bp = np.mean(background**2)

    if bp > 0:
        background *= np.sqrt(tp / (snr**2 * bp))

    mix = target + background
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix *= 0.95 / peak
        target *= 0.95 / peak

    return mix, target
