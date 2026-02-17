#!/usr/bin/env python3
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from src.separation.FastMNMF2 import FastMNMF2, MultiSTFT 

# --- Configuration ---
CONF = {
    "mix_file": "Dont_know_short.wav",
    "output_dir": "./output_separation",
    "c": 343.0,
    "n_fft": 1024,
    "hop": 256,
    "n_basis": 8,
    "n_iter": 100,
    "g_eps": 5e-2,
    "src_pos": np.array([[0.0, 6.0, 0.6], [1.0, 4.5, 0.6], [-1.0, 4.5, 0.6]]),
    "mic_pos": np.array([
        [0.0, 6.0, 1.9], [-0.5, 4.5, 1.7], [0.5, 4.5, 1.7],
        [0.0, 3.0, 1.5], [0.0, 3.0, 1.55], [0.0, 3.0, 1.45],
        [0.0, 0.0, 1.8]
    ])
}

# --- Utilitaires ---

def get_spherical_steering_vector(src, mics, freqs, c=343.0):
    dists = np.linalg.norm(mics - src, axis=1)
    phases = np.exp(-1j * 2 * np.pi * freqs[:, None] * (dists / c))
    return phases * (1.0 / (dists + 1e-3))[None, :]

def build_A_completed(conf, sr):
    n_freq = conf["n_fft"] // 2 + 1
    n_mics, n_src = len(conf["mic_pos"]), len(conf["src_pos"])
    freqs = np.linspace(0, sr/2, n_freq)
    A = np.zeros((n_freq, n_mics, n_mics), dtype=np.complex64)

    for j in range(n_src):
        a_vec = get_spherical_steering_vector(conf["src_pos"][j], conf["mic_pos"], freqs, conf["c"])
        A[:, :, j] = a_vec / (np.linalg.norm(a_vec, axis=1, keepdims=True) + 1e-10)

    for f in range(n_freq):
        U, _, _ = np.linalg.svd(A[f, :, :n_src], full_matrices=True)
        A[f, :, n_src:] = U[:, n_src:]
    return A

def plot_spatial_attention(model, save_path="spatial_attention.png"):
    """Affiche la matrice G_NM pour vérifier la cohérence spatiale."""
    # Récupération de G_NM 
    if hasattr(model.xp, 'asnumpy'):
        G = model.xp.asnumpy(model.G_NM)
    else:
        G = model.G_NM
    
    n_src, n_components = G.shape
    
    plt.figure(figsize=(10, 6))
    plt.imshow(G, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Poids (Attention)')
    
    plt.title("Matrice G_NM : Association Source vs Composantes Spatiales")
    plt.xlabel(f"Composantes Spatiales (0 à {n_src-1}=Cibles, {n_src} à {n_components-1}=Virtuelles)")
    plt.ylabel("Sources Estimées")
    
    # n_src correspond au nombre de sources définies dans le modèle
    plt.axvline(x=n_src - 0.5, color='red', linestyle='--', linewidth=2, label='Limite SVD')
    plt.legend()
    
    for i in range(n_src):
        plt.text(i, i, "★", ha="center", va="center", color="white", fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")

