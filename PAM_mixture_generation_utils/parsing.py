# =============================================================================
# METADATA PARSING
# =============================================================================


import yaml
from pathlib import Path


def load_slakh_metadata(track_dir:Path):
    meta_file = track_dir / "metadata.yaml"
    if not meta_file.exists():
        return None

    with open(meta_file, "r") as f:
        meta = yaml.safe_load(f)

    stems = {}
    for sid, s in meta["stems"].items():
        #if not s.get("audio_rendered", False):
        #    continue
        stems[sid] = {
            "instrument": s["midi_program_name"],
        }

    return {
        "track_id": track_dir.name,
        "stems": stems,
        "stems_dir": track_dir / "stems",
    }


def analyze_slakh(split, SLAKH_ROOT, INSTRUMENTS_OF_INTEREST):
    """
    Analyzes the SLakh dataset to find tracks with and without the instruments of interest.
    Args:
        split: The dataset split to analyze (e.g., "train", "validation", "test").
        SLAKH_ROOT: The root directory of the SLakh dataset.
        INSTRUMENTS_OF_INTEREST: A list of instrument names to look for in the tracks. If empty, all tracks will be considered as "with_interest".
    Returns:
        A dictionary with two keys: "with_interest" and "without_interest". Each key maps to a list of track metadata dictionaries.
    """
    split_dir = Path(SLAKH_ROOT) / split
    tracks = []

    for t in split_dir.glob("Track*"):
        meta = load_slakh_metadata(t)
        #print(meta)
        if meta is not None:
            tracks.append(meta)

    if INSTRUMENTS_OF_INTEREST:
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
    else :
        return {
            "with_interest": tracks,
            "without_interest": [],
        }