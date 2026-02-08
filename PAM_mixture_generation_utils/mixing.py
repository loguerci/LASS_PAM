# =============================================================================
# MIX CREATION
# =============================================================================


import random
from .audio import *



def create_midi_only_mix(track_meta, target_instrument, INSTRUMENT_PROMPTS:list[str]):
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


def create_midi_real_mix(real_wav, instrument, bg_track, INSTRUMENT_PROMPTS:list[str]):
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