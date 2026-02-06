"""
Create training mixtures from Slakh2100 dataset
Supports:
- Pure MIDI mixtures (Slakh only)
- Hybrid mixtures (Real recordings + Slakh MIDI backgrounds)

IMPORTANT:
- target.wav = stem used in mixture (for loss computation)
- reference.wav = different excerpt for conditioning
"""

import os
import yaml
import json
import random
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

SLAKH_ROOT = "data/raw/slakh2100_yourmt3_16k"
OUTPUT_ROOT = "data/processed"
SAMPLE_RATE = 16000

# Instruments d'intérêt (vos enregistrements réels)
INSTRUMENTS_OF_INTEREST = [
    'Saxophone',  # Saxophone ténor
    'Piano',      # Piano
    'Violin'      # Violon
]

# Mapping texte pour les prompts
INSTRUMENT_PROMPTS = {
    'Saxophone': 'saxophone tenor',
    'Piano': 'piano',
    'Violin': 'violin',
    # Autres instruments (pour MIDI uniquement)
    'Guitar': 'guitar',
    'Bass': 'bass',
    'Drums': 'drums',
    'Strings': 'strings',
    'Synth': 'synthesizer',
    'Organ': 'organ',
    'Brass': 'brass',
}

# ============================================================================
# Utilities
# ============================================================================

def load_slakh_metadata(track_dir):
    """Charge les metadata d'une piste Slakh"""
    metadata_file = track_dir / "metadata.yaml"
    
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    stems_info = {}
    if 'stems' in metadata:
        for stem_id, stem_data in metadata['stems'].items():
            inst_class = stem_data.get('inst_class', 'Unknown')
            stems_info[stem_id] = {
                'instrument': inst_class,
                'plugin_name': stem_data.get('plugin_name', ''),
                'midi_program': stem_data.get('program_num', -1)
            }
    
    return {
        'track_id': track_dir.name,
        'stems': stems_info,
        'audio_dir': track_dir / 'stems',
        'mix_file': track_dir / 'mix.wav'
    }


def get_track_instruments(metadata):
    """Liste tous les instruments dans une piste"""
    if metadata is None:
        return []
    
    instruments = []
    for stem_info in metadata['stems'].values():
        instruments.append(stem_info['instrument'])
    
    return instruments


def is_valid_background_track(metadata, exclude_instruments):
    """
    Vérifie qu'une piste Slakh ne contient AUCUN des instruments exclus
    Important: évite d'avoir 2 saxos dans le même mix (1 réel + 1 MIDI)
    """
    if metadata is None:
        return False
    
    track_instruments = get_track_instruments(metadata)
    has_excluded = any(inst in track_instruments for inst in exclude_instruments)
    
    return not has_excluded


def has_instrument(metadata, instrument_name):
    """Vérifie si une piste contient un instrument donné"""
    if metadata is None:
        return False
    
    track_instruments = get_track_instruments(metadata)
    return instrument_name in track_instruments


def load_audio_segment(audio_path, duration=10.0, offset=None, sr=16000):
    """Charge un segment audio"""
    info = sf.info(audio_path)
    total_duration = info.duration
    
    # Si trop court, charger tout et padder
    if total_duration < duration:
        audio, _ = librosa.load(audio_path, sr=sr)
        pad_length = int(duration * sr) - len(audio)
        if pad_length > 0:
            audio = np.pad(audio, (0, pad_length), mode='constant')
        return audio
    
    # Choisir un offset aléatoire si non spécifié
    if offset is None:
        max_offset = total_duration - duration
        offset = random.uniform(0, max_offset) if max_offset > 0 else 0
    
    audio, _ = librosa.load(audio_path, sr=sr, duration=duration, offset=offset)
    
    return audio


def normalize_audio(audio, target_db=-20):
    """Normalise l'audio à un niveau cible en dB"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        audio = audio * (target_rms / rms)
    return audio


def mix_with_snr(source, background, snr_db):
    """Mélange source et background avec un SNR donné"""
    source = normalize_audio(source, target_db=-20)
    background = normalize_audio(background, target_db=-20)
    
    snr_linear = 10 ** (snr_db / 20)
    source_power = np.mean(source ** 2)
    background_power = np.mean(background ** 2)
    
    if background_power > 0:
        background_scale = np.sqrt(source_power / (snr_linear ** 2 * background_power))
    else:
        background_scale = 0
    
    background_scaled = background * background_scale
    mixture = source + background_scaled
    
    # Éviter le clipping
    max_val = np.abs(mixture).max()
    if max_val > 0.95:
        scale = 0.95 / max_val
        mixture *= scale
        source *= scale
        background_scaled *= scale
    
    return mixture, source, background_scaled


# ============================================================================
# Dataset Analysis
# ============================================================================

def analyze_slakh_dataset(slakh_root, split='train'):
    """Analyse le dataset Slakh"""
    print(f"\n{'='*60}")
    print(f"Analyzing Slakh2100 {split} split")
    print(f"{'='*60}\n")
    
    split_dir = Path(slakh_root) / split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    all_tracks = sorted(split_dir.glob('Track*'))
    print(f"Found {len(all_tracks)} tracks in {split} split")
    
    valid_backgrounds = []
    tracks_with_interest = {inst: [] for inst in INSTRUMENTS_OF_INTEREST}
    tracks_without_interest = []
    instrument_counts = defaultdict(int)
    
    for track_dir in tqdm(all_tracks, desc="Analyzing tracks"):
        metadata = load_slakh_metadata(track_dir)
        
        if metadata is None:
            continue
        
        track_instruments = get_track_instruments(metadata)
        
        for inst in track_instruments:
            instrument_counts[inst] += 1
        
        # Vérifier si contient des instruments d'intérêt
        has_interest_inst = any(inst in track_instruments for inst in INSTRUMENTS_OF_INTEREST)
        
        if has_interest_inst:
            # Ajouter aux pistes avec instruments d'intérêt
            for inst in INSTRUMENTS_OF_INTEREST:
                if inst in track_instruments:
                    tracks_with_interest[inst].append(metadata)
        else:
            # Piste sans instruments d'intérêt = bon pour background
            tracks_without_interest.append(metadata)
            valid_backgrounds.append(metadata)
    
    # Afficher les statistiques
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total tracks: {len(all_tracks)}")
    print(f"\nTracks WITHOUT instruments of interest: {len(tracks_without_interest)}")
    print(f"  → Good for MIDI-only training and hybrid backgrounds")
    print(f"\nTracks WITH instruments of interest:")
    for inst in INSTRUMENTS_OF_INTEREST:
        count = len(tracks_with_interest[inst])
        print(f"  - {inst}: {count} tracks")
    
    print(f"\nTop 15 instruments:")
    for inst, count in sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        marker = "⚠️ " if inst in INSTRUMENTS_OF_INTEREST else "   "
        print(f"{marker}{inst:30s} : {count:4d} tracks")
    
    # Sauvegarder
    output_file = Path(OUTPUT_ROOT) / f"analysis_{split}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'total_tracks': len(all_tracks),
            'tracks_without_interest': len(tracks_without_interest),
            'tracks_with_interest': {k: len(v) for k, v in tracks_with_interest.items()},
            'instrument_counts': dict(instrument_counts)
        }, f, indent=2)
    
    print(f"\n✓ Saved analysis to {output_file}")
    
    return {
        'total_tracks': len(all_tracks),
        'valid_backgrounds': valid_backgrounds,
        'tracks_without_interest': tracks_without_interest,
        'tracks_with_interest': tracks_with_interest,
        'instrument_counts': dict(instrument_counts)
    }


# ============================================================================
# Mixture Creation - MIDI Only
# ============================================================================

def create_midi_mixture(
    slakh_metadata,
    target_instrument,
    duration=10.0,
    sr=16000
):
    """
    Crée un mélange MIDI uniquement
    
    CRITICAL:
    - target = stem utilisé dans le mix (pour la loss)
    - reference = AUTRE extrait du même stem (pour conditioning)
    """
    stems_dir = slakh_metadata['audio_dir']
    
    # Trouver le stem de l'instrument cible
    target_stem_id = None
    for stem_id, stem_info in slakh_metadata['stems'].items():
        if stem_info['instrument'] == target_instrument:
            target_stem_id = stem_id
            break
    
    if target_stem_id is None:
        return None
    
    target_file = stems_dir / f"{target_stem_id}.wav"
    if not target_file.exists():
        return None
    
    # Durée totale du fichier
    info = sf.info(target_file)
    total_duration = info.duration
    
    if total_duration < duration * 2:
        # Fichier trop court pour avoir target ET reference différents
        return None
    
    # ─────────────────────────────────────────────────────────────
    # TARGET: extrait utilisé dans le mix
    # ─────────────────────────────────────────────────────────────
    target_offset = random.uniform(0, total_duration - duration * 2)
    target_audio = load_audio_segment(target_file, duration=duration, offset=target_offset, sr=sr)
    
    # ─────────────────────────────────────────────────────────────
    # REFERENCE: AUTRE extrait du même instrument (pour conditioning)
    # ─────────────────────────────────────────────────────────────
    # Prendre un extrait différent (au moins 5 secondes d'écart)
    ref_offset = target_offset + duration + random.uniform(5, total_duration - target_offset - duration * 2)
    if ref_offset + duration > total_duration:
        ref_offset = random.uniform(0, target_offset - duration) if target_offset > duration else 0
    
    reference_audio = load_audio_segment(target_file, duration=duration, offset=ref_offset, sr=sr)
    
    # ─────────────────────────────────────────────────────────────
    # BACKGROUND: tous les autres stems
    # ─────────────────────────────────────────────────────────────
    background_audio = np.zeros_like(target_audio)
    background_stems = []
    
    for stem_id, stem_info in slakh_metadata['stems'].items():
        if stem_id == target_stem_id:
            continue
        
        stem_file = stems_dir / f"{stem_id}.wav"
        if stem_file.exists():
            stem_audio = load_audio_segment(stem_file, duration=duration, offset=target_offset, sr=sr)
            background_audio += stem_audio
            background_stems.append(stem_info['instrument'])
    
    # ─────────────────────────────────────────────────────────────
    # MIX
    # ─────────────────────────────────────────────────────────────
    snr = random.uniform(-2, 8)  # dB
    mixture, target_scaled, background_scaled = mix_with_snr(
        target_audio, 
        background_audio, 
        snr
    )
    
    prompt = INSTRUMENT_PROMPTS.get(target_instrument, target_instrument.lower())
    
    return {
        'mixture': mixture,
        'target': target_scaled,        # Pour la loss
        'reference': reference_audio,   # Pour le conditioning
        'prompt': prompt,
        'metadata': {
            'type': 'midi',
            'source_track': slakh_metadata['track_id'],
            'target_instrument': target_instrument,
            'target_stem': target_stem_id,
            'background_instruments': background_stems,
            'snr_db': snr,
            'duration': duration,
            'target_offset': target_offset,
            'reference_offset': ref_offset
        }
    }


# ============================================================================
# Mixture Creation - Hybrid (Real + MIDI)
# ============================================================================

def create_hybrid_mixture(
    real_recording_path,
    slakh_background_metadata,
    instrument_name,
    duration=10.0,
    sr=16000
):
    """
    Crée un mélange hybride : enregistrement réel + background MIDI Slakh
    
    CRITICAL:
    - target = extrait de l'enregistrement réel utilisé dans le mix
    - reference = AUTRE extrait du même enregistrement réel
    """
    if not Path(real_recording_path).exists():
        return None
    
    info = sf.info(real_recording_path)
    total_duration = info.duration
    
    if total_duration < duration * 2:
        # Fichier trop court
        return None
    
    # ─────────────────────────────────────────────────────────────
    # TARGET: extrait réel utilisé dans le mix
    # ─────────────────────────────────────────────────────────────
    target_offset = random.uniform(0, total_duration - duration * 2)
    target_audio = load_audio_segment(real_recording_path, duration=duration, offset=target_offset, sr=sr)
    
    # ─────────────────────────────────────────────────────────────
    # REFERENCE: AUTRE extrait du même enregistrement
    # ─────────────────────────────────────────────────────────────
    ref_offset = target_offset + duration + random.uniform(5, min(20, total_duration - target_offset - duration * 2))
    if ref_offset + duration > total_duration:
        ref_offset = max(0, target_offset - duration - 5)
    
    reference_audio = load_audio_segment(real_recording_path, duration=duration, offset=ref_offset, sr=sr)
    
    # ─────────────────────────────────────────────────────────────
    # BACKGROUND: stems Slakh (sans instruments d'intérêt)
    # ─────────────────────────────────────────────────────────────
    background_audio = np.zeros_like(target_audio)
    background_instruments = []
    
    stems_dir = slakh_background_metadata['audio_dir']
    background_offset = random.uniform(0, 60)  # Offset aléatoire pour le background
    
    for stem_id, stem_info in slakh_background_metadata['stems'].items():
        # Vérifier que ce n'est pas un instrument d'intérêt
        if stem_info['instrument'] in INSTRUMENTS_OF_INTEREST:
            continue
        
        stem_file = stems_dir / f"{stem_id}.wav"
        
        if stem_file.exists():
            try:
                stem_audio = load_audio_segment(stem_file, duration=duration, offset=background_offset, sr=sr)
                background_audio += stem_audio
                background_instruments.append(stem_info['instrument'])
            except:
                continue
    
    # ─────────────────────────────────────────────────────────────
    # MIX
    # ─────────────────────────────────────────────────────────────
    snr = random.uniform(0, 10)  # dB
    mixture, target_scaled, background_scaled = mix_with_snr(
        target_audio,
        background_audio,
        snr
    )
    
    prompt = INSTRUMENT_PROMPTS.get(instrument_name, instrument_name.lower())
    
    return {
        'mixture': mixture,
        'target': target_scaled,        # Pour la loss
        'reference': reference_audio,   # Pour le conditioning
        'prompt': prompt,
        'metadata': {
            'type': 'hybrid',
            'real_recording': str(real_recording_path),
            'slakh_background': slakh_background_metadata['track_id'],
            'target_instrument': instrument_name,
            'background_instruments': background_instruments,
            'snr_db': snr,
            'duration': duration,
            'target_offset': target_offset,
            'reference_offset': ref_offset,
            'background_offset': background_offset
        }
    }


# ============================================================================
# Save Mixture
# ============================================================================

def save_mixture(mixture_data, output_dir, example_id):
    """Sauvegarde un exemple de mélange"""
    example_dir = Path(output_dir) / f"example_{example_id:05d}"
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les fichiers audio
    sf.write(example_dir / 'mixture.wav', mixture_data['mixture'], SAMPLE_RATE)
    sf.write(example_dir / 'target.wav', mixture_data['target'], SAMPLE_RATE)
    sf.write(example_dir / 'reference.wav', mixture_data['reference'], SAMPLE_RATE)
    
    # Sauvegarder les metadata
    metadata = {
        'prompt': mixture_data['prompt'],
        **mixture_data['metadata']
    }
    
    with open(example_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_midi_dataset(
    analysis_results,
    num_examples=200,
    output_dir=None,
    duration=10.0,
    exclude_interest_instruments=True
):
    """
    Génère un dataset MIDI uniquement
    
    Args:
        exclude_interest_instruments: Si True, n'utilise PAS saxo/piano/violon MIDI
                                      (recommandé pour éviter confusion)
    """
    if output_dir is None:
        output_dir = Path(OUTPUT_ROOT) / 'train_midi'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating MIDI-only dataset")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Target examples: {num_examples}")
    print(f"Exclude instruments of interest: {exclude_interest_instruments}")
    print()
    
    # Choisir les pistes à utiliser
    if exclude_interest_instruments:
        # Utiliser uniquement les pistes SANS saxo/piano/violon
        available_tracks = analysis_results['tracks_without_interest']
        print(f"Using {len(available_tracks)} tracks without instruments of interest")
    else:
        # Utiliser toutes les pistes
        available_tracks = []
        for track_list in analysis_results['tracks_with_interest'].values():
            available_tracks.extend(track_list)
        print(f"Using all tracks (including instruments of interest)")
    
    if len(available_tracks) == 0:
        print("Error: No available tracks!")
        return 0
    
    examples_created = 0
    attempts = 0
    max_attempts = num_examples * 10
    
    with tqdm(total=num_examples, desc="Creating MIDI mixtures") as pbar:
        while examples_created < num_examples and attempts < max_attempts:
            attempts += 1
            
            # Choisir une piste aléatoire
            track_metadata = random.choice(available_tracks)
            
            # Choisir un instrument aléatoire dans cette piste
            track_instruments = list(track_metadata['stems'].values())
            if len(track_instruments) == 0:
                continue
            
            target_stem = random.choice(track_instruments)
            target_instrument = target_stem['instrument']
            
            # Skip si pas dans INSTRUMENT_PROMPTS
            if target_instrument not in INSTRUMENT_PROMPTS:
                continue
            
            # Créer le mélange
            mixture_data = create_midi_mixture(
                track_metadata,
                target_instrument,
                duration=duration
            )
            
            if mixture_data is not None:
                save_mixture(mixture_data, output_dir, examples_created)
                examples_created += 1
                pbar.update(1)
    
    print(f"\n✓ Created {examples_created} MIDI examples")
    print(f"  (took {attempts} attempts)")
    
    # Statistiques
    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'total_examples': examples_created,
            'duration': duration,
            'sample_rate': SAMPLE_RATE,
            'type': 'midi',
            'exclude_interest_instruments': exclude_interest_instruments
        }, f, indent=2)
    
    return examples_created


def generate_hybrid_dataset(
    real_recordings_dir,
    analysis_results,
    num_examples_per_instrument=50,
    output_dir=None,
    duration=10.0
):
    """Génère un dataset hybride"""
    if output_dir is None:
        output_dir = Path(OUTPUT_ROOT) / 'train_hybrid'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    real_recordings_dir = Path(real_recordings_dir)
    
    if not real_recordings_dir.exists():
        print(f"\nWarning: Real recordings directory not found: {real_recordings_dir}")
        print("Please record your musicians first!")
        return 0
    
    print(f"\n{'='*60}")
    print(f"Generating Hybrid dataset (Real + MIDI)")
    print(f"{'='*60}")
    print(f"Real recordings: {real_recordings_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Examples per instrument: {num_examples_per_instrument}")
    print()
    
    valid_backgrounds = analysis_results['valid_backgrounds']
    
    if len(valid_backgrounds) == 0:
        print("Error: No valid background tracks!")
        return 0
    
    examples_created = 0
    
    # Pour chaque instrument d'intérêt
    for instrument in INSTRUMENTS_OF_INTEREST:
        inst_dir = real_recordings_dir / instrument.lower()
        
        if not inst_dir.exists():
            print(f"Skipping {instrument}: directory not found")
            continue
        
        # Trouver tous les enregistrements réels
        real_files = list(inst_dir.glob('**/*.wav'))
        
        if len(real_files) == 0:
            print(f"Skipping {instrument}: no .wav files found")
            continue
        
        print(f"\n{instrument}: {len(real_files)} recordings found")
        
        # Créer des mélanges
        for i in tqdm(range(num_examples_per_instrument), desc=f"  {instrument}"):
            # Choisir un enregistrement réel aléatoire
            real_file = random.choice(real_files)
            
            # Choisir un background Slakh aléatoire (SANS cet instrument)
            background_metadata = random.choice(valid_backgrounds)
            
            # Créer le mélange
            mixture_data = create_hybrid_mixture(
                real_file,
                background_metadata,
                instrument,
                duration=duration
            )
            
            if mixture_data is not None:
                save_mixture(mixture_data, output_dir, examples_created)
                examples_created += 1
    
    print(f"\n✓ Created {examples_created} hybrid examples")
    
    # Statistiques
    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'total_examples': examples_created,
            'duration': duration,
            'sample_rate': SAMPLE_RATE,
            'type': 'hybrid',
            'real_recordings_dir': str(real_recordings_dir)
        }, f, indent=2)
    
    return examples_created


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create training mixtures from Slakh2100')
    parser.add_argument('--mode', choices=['analyze', 'midi', 'hybrid', 'all'], 
                        default='all', help='Operation mode')
    parser.add_argument('--split', default='train', 
                        choices=['train', 'validation', 'test'],
                        help='Slakh split to use')
    parser.add_argument('--num-midi', type=int, default=200,
                        help='Number of MIDI examples to create')
    parser.add_argument('--exclude-interest', action='store_true', default=True,
                        help='Exclude instruments of interest from MIDI dataset (recommended)')
    parser.add_argument('--num-hybrid-per-inst', type=int, default=50,
                        help='Number of hybrid examples per instrument')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration of each mixture (seconds)')
    parser.add_argument('--real-recordings', type=str, 
                        default='data/solo_recordings',
                        help='Directory containing real recordings')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Slakh2100 Mixture Creation")
    print(f"{'='*60}\n")
    
    # Analyze dataset
    if args.mode in ['analyze', 'all']:
        analysis_results = analyze_slakh_dataset(SLAKH_ROOT, args.split)
    else:
        # Load previous analysis
        analysis_file = Path(OUTPUT_ROOT) / f"analysis_{args.split}.json"
        if not analysis_file.exists():
            print("Please run with --mode analyze first!")
            return
        
        # Reload (simplifié - vous pourriez cacher les metadata)
        print("Loading previous analysis...")
        analysis_results = analyze_slakh_dataset(SLAKH_ROOT, args.split)
    
    # Generate MIDI dataset
    if args.mode in ['midi', 'all']:
        generate_midi_dataset(
            analysis_results,
            num_examples=args.num_midi,
            duration=args.duration,
            exclude_interest_instruments=args.exclude_interest
        )
    
    # Generate hybrid dataset
    if args.mode in ['hybrid', 'all']:
        generate_hybrid_dataset(
            args.real_recordings,
            analysis_results,
            num_examples_per_instrument=args.num_hybrid_per_inst,
            duration=args.duration
        )
    
    print(f"\n{'='*60}")
    print("✓ Dataset creation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()