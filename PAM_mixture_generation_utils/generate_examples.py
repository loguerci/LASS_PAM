import audio as aud
import parsing
from IPython.display import display, Audio
import os
import numpy as np
from pathlib import Path
import random
import json

CLIP_LENGTH = 10
SAMPLE_RATE = 16000
NUMBER_OF_DATAPOINTS = -1

DATASET_DIR = Path("<COMPLETE HERE>/slakh2100_yourmt3_16k/train")
RECORDED_DIR = Path("<COMPLETE HERE>/Records/Records")
OUTPUT_DIR = Path("<COMPLETE HERE>/generated_dataset")

REFERENCE_IS_CLIP_LENGTH = True
VERBOSE = True

FILTER = ["violin", "piano", "sax", "violon_rec", "piano_rec", "saxophone_rec"]
REJECT = ["electric", "synth"]




assert "COMPLETE HERE" not in str(DATASET_DIR), "Please set the DATASET_DIR variable to the path of your dataset directory"
assert "COMPLETE HERE" not in str(RECORDED_DIR), "Please set the RECORDED_DIR variable to the path of your recorded directory"
assert "COMPLETE HERE" not in str(OUTPUT_DIR), "Please set the OUTPUT_DIR variable to the path of your desired output directory"
assert NUMBER_OF_DATAPOINTS >= 0, "Please set the NUMBER_OF_DATAPOINTS variable to a non-negative integer. 0 will skip generation."





def instrument_sets(dataset_dir : Path = DATASET_DIR, recorded_dir : Path = RECORDED_DIR) -> dict[str, list[Path]]:
    instr_sets = dict()
    for f in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, f)):
            print(f"{f} is not a directory, skipping")
            continue
        meta = parsing.track_metadata(os.path.join(dataset_dir,f))
        instrs = parsing.track_stems_and_instr(meta)
        for s, i in instrs.items():
            if i not in instr_sets :
                instr_sets[i] = list()
            instr_sets[i].append(Path(os.path.join(dataset_dir, f + '/stems/' + s + '.wav')))
    
    # recording integration
    keys = ["violon", "saxophone", "piano"]
    for dir in os.listdir(recorded_dir):
        if not os.path.isdir(os.path.join(recorded_dir, dir)):
            print(f"{f} is not a directory, skipping")
            continue
        if dir in ["1. I will survive", "2. Hey Jude", "3. Don't know why"]:
            print(f"{f}  is a forbidden directory, skipping")
            continue
        print(recorded_dir,  dir)
        print(recorded_dir + '/' + dir)
        files = list(os.listdir(recorded_dir +'/' + dir))

        for f in files:
            for k in keys:
                k_instr = k + "_rec"
                if k.upper() in f.upper():
                    if k_instr not in instr_sets:
                        instr_sets[k_instr] = list()
                    instr_sets[k_instr].append(recorded_dir + '/' + dir + '/' + f)
    return instr_sets

def filter_instr_by_keywords(instr_sets:dict[str, list[Path]], filt, rej, recorded_dir : Path = RECORDED_DIR) -> dict[str, list[Path]]:
    res = dict()
    for f in filt:
        if f not in res:
            res[f] = list()
        print(instr_sets)
        for inst, stems in instr_sets.items():
            if f.upper() in inst.upper() and not any(r.upper() in inst.upper() for r in rej):
                res[f] += stems
    return res

def chose_rand_stems(instrs_and_stems : dict[str, list[Path]], instruments : list[str], n=1) -> list[tuple[str, str, Path]]:
    """ returns a list of n tuples of the form (instrument type (keyword), instrument, stem_path)
        for each keyword in `instruments`,
        where instrument is a random instrument matching the keyword
        stem_path is a random stem path for that instrument. """
    matches = {i: [k for k in instrs_and_stems if i.upper() in k.upper()] for i in instruments}
    res = list()
    for i in instruments:
        for _ in range(n):
            k = random.choice(matches[i])
            stem_path = random.choice(instrs_and_stems[k])
            res.append((i, k, stem_path))
    return res

def build_datapoint(instrs_and_stems : dict[str, list[Path]], filtered_instr_and_stems : dict[str, list[Path]], prompt_target_recording:tuple[str, str, Path], mix_duration_s=30, mix_division=5, instance_probability=.5, max_seg_duration_s=5.0):
    res = dict()
    metadata = dict()
    background_instruments = set()
    while len(background_instruments) < 4:
        attempt = None
        while attempt is None:
            b = random.choice(list(instrs_and_stems.keys()))
            if prompt_target_recording[1].upper() not in b.upper():
                attempt = aud.load_audio_segment(random.choice(instrs_and_stems[b]), convolve_length=256)
                if attempt is None:
                    print(f"Failed to load background instrument {b}, retrying...")
        background_instruments.add(b)
    background_keyword_instrument_stems = chose_rand_stems(instrs_and_stems, list(background_instruments), n=1)

    _prompt_instr = [i for i in filtered_instr_and_stems.keys() if prompt_target_recording[0].upper() in i.upper()]
    reference = None
    for i in range(10):
        x = filtered_instr_and_stems[random.choice(_prompt_instr)]
        if not x:
            continue
        reference = aud.normalize_energy(aud.load_audio_segment(random.choice(x), convolve_length=256))
        if reference is not None:
            break
    if reference is None:
        raise Exception(f"Could not load reference audio for prompt {prompt_target_recording[0]} after 10 attempts")
    if REFERENCE_IS_CLIP_LENGTH:
        reference = reference[:min(CLIP_LENGTH * SAMPLE_RATE, len(reference))]
    
    target_raw = None
    for i in range(10):
        target_raw = aud.normalize_energy(aud.load_audio_segment(prompt_target_recording[2], convolve_length=256))
        if target_raw is not None:
            break
    if target_raw is None:
        raise Exception(f"Could not load target audio for prompt {prompt_target_recording[1]} after 10 attempts")


    mixture = aud.scatter_audio_segments([aud.load_audio_segment(b[2], convolve_length=256) for b in background_keyword_instrument_stems], mix_duration_s=mix_duration_s, mix_division=mix_division, instance_probability=instance_probability, max_seg_duration_s=max_seg_duration_s)

    res['reference'] = reference
    res['target'] = aud.scatter_audio_segments([target_raw], mix_duration_s=mix_duration_s, mix_division=mix_division, instance_probability=instance_probability, max_seg_duration_s=max_seg_duration_s, dbg = True)
    mixture += np.nan_to_num(res['target'], nan=0.0, posinf=0.0, neginf=0.0)
    mixture = aud.normalize_energy(mixture, alpha=.9)
    res['mixture'] = mixture
    
    metadata['prompt'] = prompt_target_recording[0]
    metadata['target_instrument'] = prompt_target_recording[1]
    metadata['background_instruments'] = [b[1] for b in background_keyword_instrument_stems]

    meta_json = json.dumps(metadata)
    return res, meta_json, background_instruments

def save_datapoint(datapoint : dict[str, np.ndarray], metadata_json : str, save_dir : str):
    os.makedirs(save_dir, exist_ok=True)
    aud.save_audio(os.path.join(save_dir, 'reference.wav'), datapoint['reference'])
    aud.save_audio(os.path.join(save_dir, 'target.wav'), datapoint['target'])
    aud.save_audio(os.path.join(save_dir, 'mixture.wav'), datapoint['mixture'])
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        f.write(metadata_json)


print("loading...")
instrs_and_stems = instrument_sets(DATASET_DIR, recorded_dir=RECORDED_DIR)
if VERBOSE:
    print("-"*20 + "INSTRUMENTS" + "-"*20)
    for k, v in instrs_and_stems.items():
        print(f"    {k}: {v}")

print("filtering instruments by keywords...")
filtered = filter_instr_by_keywords(instrs_and_stems, FILTER, REJECT, recorded_dir=RECORDED_DIR)
if VERBOSE:
    print("-"*20 + "POTENTIAL TARGETS" + "-"*20)
    for k, v in instrs_and_stems.items():
        print(f"    {k}: {v}")

print("generating datapoints...")
for i in range(NUMBER_OF_DATAPOINTS):
    attempt = None
    while attempt is None:
        prompt_target = random.choice(chose_rand_stems(filtered, FILTER, n=1))
        p, inst, t = prompt_target
        if p == "saxophone_rec":
           p = "saxophone"
        if p == "violon_rec":
            p = "violin"
        if p == "piano_rec":
            p = "piano"
        prompt_target  = (p, inst, t)

        attempt = aud.load_audio_segment(prompt_target[2], convolve_length=256)
        if attempt is None and VERBOSE:
             print(f"Failed to load prompt target {prompt_target[2]}, retrying...")
    
    datapoint, metadata_json, background = build_datapoint(instrs_and_stems, filtered, prompt_target, mix_duration_s=10, mix_division=5, instance_probability=.8, max_seg_duration_s=3.0)
    
    if VERBOSE:
        print("-"*20 + f"DATAPOINT {i} ({100*(i+1)/NUMBER_OF_DATAPOINTS:.0f}%)" + "-"*20)
        print("prompt, target and stem :", *prompt_target)
        print("background stems :", background)
    else:
        print(f"{100*(i+1)/NUMBER_OF_DATAPOINTS:.0f}%")
    print(metadata_json.replace(",", ",\n"))
    
    save_datapoint(datapoint, metadata_json, os.path.join(OUTPUT_DIR, f"example_{i}"))

    print(f"saved to {os.path.join(OUTPUT_DIR, f'example_{i}')}")
    print("-"*50)