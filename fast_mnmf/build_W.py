import os
import numpy as np
import librosa
from tqdm import tqdm



SR = 44100
N_FFT = 2048
HOP = 512

DATASET_PATH = "..\W_dataset"
OUTPUT_PATH = "W_dictionary.npy"

SOURCES = ["piano", "violon", "sax"]


def extract_basis(wav_path):
    y, sr = librosa.load(wav_path, sr=SR)

    # STFT
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    V = np.abs(S) ** 2

    # Moyenne temporelle -> signature spectrale
    w = np.mean(V, axis=1)

    # Normalisation (TRÈS IMPORTANT)
    w = w / (np.sum(w) + 1e-12)

    return w



all_W = []
basis_count = []

for src in SOURCES:

    src_path = os.path.join(DATASET_PATH, src)
    files = [f for f in os.listdir(src_path) if f.endswith(".wav")]

    W_src = []

    print(f"\nProcessing {src}")

    for f in tqdm(files):
        path = os.path.join(src_path, f)
        w = extract_basis(path)
        W_src.append(w)

    W_src = np.stack(W_src, axis=1)  # (F, K)
    all_W.append(W_src)
    basis_count.append(W_src.shape[1])



F = all_W[0].shape[0]
Kmax = max(basis_count)
N = len(SOURCES)

W_NFK = np.zeros((N, F, Kmax))

for n, W in enumerate(all_W):
    K = W.shape[1]
    W_NFK[n, :, :K] = W



np.save(OUTPUT_PATH, W_NFK)

print("\nDONE")
print("Shape W_NFK =", W_NFK.shape)
print("Saved to", OUTPUT_PATH)
