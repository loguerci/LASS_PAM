"""
Mix creation utilities
"""
import random
from pathlib import Path
from .audio import *


def create_midi_only_mix(track_meta, target_instrument:(None | str), INSTRUMENT_PROMPTS):
    """
    Create MIDI-only mixture: 1 target stem + 1 background stem
    
    Returns None if unable to create valid mixture
    """
    stems = track_meta["stems"]
    stems_dir = track_meta["stems_dir"]

    # Find target stem
    if target_instrument is None:
        target_instrument = random.choice(list(stems.values()))["instrument"]
    target_ids = [
        sid for sid, s in stems.items()
        if s["instrument"] == target_instrument
    ]
    if not target_ids:
        return None

    target_sid = random.choice(target_ids)
    target_file = stems_dir / f"{target_sid}.wav"
    
    if not target_file.exists():
        return None

    # Load target and reference (two different segments)
    target_audio, reference_audio = load_two_different_segments(target_file)
    
    if target_audio is None or reference_audio is None:
        return None

    # Find ONE background stem (different instrument)
    bg_candidates = [
        (sid, s) for sid, s in stems.items()
        if sid != target_sid and s["instrument"] != target_instrument
    ]
    
    if not bg_candidates:
        return None
    
    # Shuffle and try candidates
    random.shuffle(bg_candidates)
    
    for bg_sid, bg_stem in bg_candidates:
        bg_file = stems_dir / f"{bg_sid}.wav"
        if not bg_file.exists():
            continue
        
        bg_audio = load_audio_segment(bg_file)
        
        if bg_audio is None:
            continue
        
        # Try to mix
        mixture, target_scaled, _ = mix_with_snr(target_audio, bg_audio)
        
        if mixture is None:
            continue
        
        # Success!
        return {
            "mixture": mixture,
            "target": target_scaled,
            "reference": reference_audio,
            "prompt": INSTRUMENT_PROMPTS.get(target_instrument, target_instrument.lower()),
            "metadata": {
                "type": "midi_only",
                "target_instrument": target_instrument,
                "background_instrument": bg_stem["instrument"],
            }
        }
    
    # Failed to create valid mix
    return None


def create_midi_real_mix(real_wav, instrument, bg_track, INSTRUMENT_PROMPTS):
    """
    Create hybrid mixture: 1 real recording + 1 MIDI background stem
    
    Returns None if unable to create valid mixture
    """
    # Load real recording (target and reference)
    target_audio, reference_audio = load_two_different_segments(real_wav)
    
    if target_audio is None or reference_audio is None:
        return None

    # Find ONE background stem (different instrument than real)
    bg_candidates = [
        (sid, s) for sid, s in bg_track["stems"].items()
        if s["instrument"] != instrument  # Avoid same instrument
    ]
    
    if not bg_candidates:
        return None
    
    # Shuffle and try candidates
    random.shuffle(bg_candidates)
    
    for bg_sid, bg_stem in bg_candidates:
        bg_file = bg_track["stems_dir"] / f"{bg_sid}.wav"
        if not bg_file.exists():
            continue
        
        bg_audio = load_audio_segment(bg_file)
        
        if bg_audio is None:
            continue
        
        # Try to mix
        mixture, target_scaled, _ = mix_with_snr(
            target_audio, 
            bg_audio,
            snr_db_low=0,   # Real recordings: higher SNR
            snr_db_high=10
        )
        
        if mixture is None:
            continue
        
        # Success!
        return {
            "mixture": mixture,
            "target": target_scaled,
            "reference": reference_audio,
            "prompt": INSTRUMENT_PROMPTS.get(instrument, instrument.lower()),
            "metadata": {
                "type": "midi_real",
                "target_instrument": instrument,
                "background_instrument": bg_stem["instrument"],
            }
        }
    
    # Failed to create valid mix
    return None