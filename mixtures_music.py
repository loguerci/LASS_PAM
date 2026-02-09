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



from typing import Literal
import yaml
import json
import random
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from PAM_mixture_generation_utils.audio import *
from PAM_mixture_generation_utils.parsing import *
from PAM_mixture_generation_utils.mixing import *




# =============================================================================
# CONFIG
# =============================================================================

class DATASET_T(Literal):
    babyslakh_16k = "babyslakh_16k"
    # ... à compléter pour d'autres datasets
DATASET = DATASET_T.babyslakh_16k

OUTPUT_ROOT = "data/processed/slakh_mixtures"

SAMPLE_RATE = 16000
MIX_DURATION = 10.0

if DATASET == "babyslakh_16k":
    SLAKH_ROOT = "data/raw/babyslakh_16k"
    INSTRUMENTS_OF_INTEREST = []
else:
    SLAKH_ROOT = "data/raw/..." # à compléter
    INSTRUMENTS_OF_INTEREST = [
        "Tenor Sax",
        "Alto Sax",
        "Acoustic Grand Piano",
        "Bright Acoustic Piano",
        "Clavinet",
        "Honky-tonk Piano",
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


    if DATASET == "babyslakh_16k": # babyslakh_16k est trop petit pour faire les fines bouches sur les instruments et le split
        analysis = analyze_slakh('', SLAKH_ROOT, INSTRUMENTS_OF_INTEREST)
    else :
        analysis = analyze_slakh(args.split, SLAKH_ROOT, INSTRUMENTS_OF_INTEREST)


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
            instrument = random.choice(INSTRUMENTS_OF_INTEREST) if INSTRUMENTS_OF_INTEREST else random.choice(list(track["stems"].values()))["instrument"]

            ex = create_midi_only_mix(track, instrument)
            if ex is None:
                print(f"ex is None for track {track['track_id']} and instrument {instrument}, continuing...")
                continue

            save_example(ex, out_dir, examples)
            examples += 1
            pbar.update(1)

    print(f"\n✓ Created {examples} examples")


if __name__ == "__main__":
    main()
