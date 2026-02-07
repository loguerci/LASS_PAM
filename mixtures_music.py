"""
Create training mixtures from BabySlakh (16kHz)

Outputs (per example):
- mixture.wav   (10s)
- target.wav    (10s)  -> stem cible (loss)
- reference.wav (10s)  -> autre extrait du même instrument (conditioning)
- metadata.json

Supports:
- MIDI-only mixtures
- MIDI + real mixtures (future use)

Author: LASS project
"""

import yaml
import json
import random
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================

SLAKH_ROOT = "data/raw/slakh2100_yourmt3_16k"
OUTPUT_ROOT = "data/processed/slakh_mixtures"

SAMPLE_RATE = 16000
MIX_DURATION = 10.0

INSTRUMENTS_OF_INTEREST = [
    "Tenor Sax",
    "Alto Sax",
    "Acoustic Grand Piano",
    "Bright Acoustic Piano",
    "Clavinet",
    "Honky-tonk Piano"
    "Violin",
    "Viola"
]

INSTRUMENT_PROMPTS = {
    "Tenor Sax": "saxophone",
    "Alto Sax": "saxophone",
    "Acoustic Grand Piano": "piano",
    "Bright Acoustic Piano": "piano",
    "Clavinet": "piano",
    "Honky-tonk Piano": "piano",
    "Violin": "violin",
    "Viola": "viola",
}

# =============================================================================
# AUDIO UTILS
# =============================================================================

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

# =============================================================================
# METADATA
# =============================================================================

def load_slakh_metadata(track_dir):
    meta_file = track_dir / "metadata.yaml"
    if not meta_file.exists():
        return None

    with open(meta_file, "r") as f:
        meta = yaml.safe_load(f)

    stems = {}
    for sid, s in meta["stems"].items():
        if not s.get("audio_rendered", False):
            continue
        stems[sid] = {
            "instrument": s["midi_program_name"],
        }

    return {
        "track_id": track_dir.name,
        "stems": stems,
        "stems_dir": track_dir / "stems",
    }


def analyze_slakh(split):
    split_dir = Path(SLAKH_ROOT) / split
    tracks = []

    for t in split_dir.glob("Track*"):
        meta = load_slakh_metadata(t)
        if meta is not None:
            tracks.append(meta)

    with_interest = []
    without_interest = []

    for m in tracks:
        instruments = {s["instrument"] for s in m["stems"].values()}
        if any(i in instruments for i in INSTRUMENTS_OF_INTEREST):
            with_interest.append(m)
        else:
            without_interest.append(m)

    return {
        "with_interest": with_interest,
        "without_interest": without_interest,
    }

# =============================================================================
# MIX CREATION
# =============================================================================

def create_midi_only_mix(track_meta, target_instrument):
    stems = track_meta["stems"]
    stems_dir = track_meta["stems_dir"]

    target_ids = [
        sid for sid, s in stems.items()
        if s["instrument"] == target_instrument
    ]
    if not target_ids:
        return None

    target_sid = random.choice(target_ids)
    target_file = stems_dir / f"{target_sid}.wav"

    bg_audio = None
    bg_instruments = set()

    for sid, s in stems.items():
        inst = s["instrument"]
        if sid == target_sid:
            continue
        if inst == target_instrument or inst in bg_instruments:
            continue

        audio = load_audio_segment(
            stems_dir / f"{sid}.wav",
            MIX_DURATION
        )
        bg_audio = audio if bg_audio is None else bg_audio + audio
        bg_instruments.add(inst)

        if len(bg_instruments) == 2:
            break

    if bg_audio is None:
        return None

    target_audio = load_audio_segment(target_file, MIX_DURATION)
    reference_audio = load_audio_segment(target_file, MIX_DURATION)

    mixture, target_scaled = mix_with_snr(
        target_audio, bg_audio, snr_db=random.uniform(-2, 8)
    )

    return {
        "mixture": mixture,
        "target": target_scaled,
        "reference": reference_audio,
        "prompt": INSTRUMENT_PROMPTS[target_instrument],
        "metadata": {
            "type": "midi_only",
            "target_instrument": target_instrument,
            "background_instruments": list(bg_instruments),
        }
    }


def create_midi_real_mix(real_wav, instrument, bg_track):
    target_audio = load_audio_segment(real_wav, MIX_DURATION)
    reference_audio = load_audio_segment(real_wav, MIX_DURATION)

    bg_audio = None
    bg_instruments = set()

    for sid, s in bg_track["stems"].items():
        inst = s["instrument"]
        if inst == instrument or inst in bg_instruments:
            continue

        audio = load_audio_segment(
            bg_track["stems_dir"] / f"{sid}.wav",
            MIX_DURATION
        )
        bg_audio = audio if bg_audio is None else bg_audio + audio
        bg_instruments.add(inst)

        if len(bg_instruments) == 3:
            break

    if bg_audio is None:
        return None

    mixture, target_scaled = mix_with_snr(
        target_audio, bg_audio, snr_db=random.uniform(0, 10)
    )

    return {
        "mixture": mixture,
        "target": target_scaled,
        "reference": reference_audio,
        "prompt": INSTRUMENT_PROMPTS[instrument],
        "metadata": {
            "type": "midi_real",
            "target_instrument": instrument,
            "background_instruments": list(bg_instruments),
        }
    }

# =============================================================================
# SAVE
# =============================================================================

def save_example(example, out_dir, idx):
    ex_dir = out_dir / f"example_{idx:05d}"
    ex_dir.mkdir(parents=True, exist_ok=True)

    sf.write(ex_dir / "mixture.wav", example["mixture"], SAMPLE_RATE)
    sf.write(ex_dir / "target.wav", example["target"], SAMPLE_RATE)
    sf.write(ex_dir / "reference.wav", example["reference"], SAMPLE_RATE)

    with open(ex_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "prompt": example["prompt"],
                **example["metadata"]
            },
            f,
            indent=2
        )

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-midi", type=int, default=500)
    parser.add_argument("--mode", choices=["midi"], default="midi")
    args = parser.parse_args()

    analysis = analyze_slakh(args.split)
    out_dir = Path(OUTPUT_ROOT) / "midi_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = 0
    attempts = 0
    max_attempts = args.num_midi * 10

    print(f"\nCreating MIDI-only dataset ({args.num_midi} examples)\n")

    with tqdm(total=args.num_midi) as pbar:
        while examples < args.num_midi and attempts < max_attempts:
            attempts += 1

            track = random.choice(analysis["with_interest"])
            instrument = random.choice(INSTRUMENTS_OF_INTEREST)

            ex = create_midi_only_mix(track, instrument)
            if ex is None:
                continue

            save_example(ex, out_dir, examples)
            examples += 1
            pbar.update(1)

    print(f"\n✓ Created {examples} examples")


if __name__ == "__main__":
    main()
