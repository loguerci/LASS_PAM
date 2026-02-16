#---------------------------------------------------------------------------
# Process one audio file for qualitative analysis
#---------------------------------------------------------------------------
import torch
import numpy as np
from model.LASS_clap import LASS_clap
from utils.stft import STFT
import librosa
import os  

def normalize_audio(audio):
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return np.clip(audio, -1.0, 1.0)

device = torch.device('cuda')
    
stft = STFT(
    filter_length=1024,
    hop_length=256,
    win_length=1024
).to(device)
    
model = LASS_clap(device=device).to(device)
model.clap_conditioner.use_audio = False  
    
ckpt_path = '/home/infres/lgosselin-25/LASS_PAM/checkpoints/exp_lr_10e-4_best_0.24/best.pth'
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

audio_path = 'examples/Dont_know_short.wav'
mixture, sr = librosa.load(audio_path, sr=16000, mono=True)

# Split into two 10-second segments
mixture_1 = mixture[:160000]   # First 10 seconds
mixture_2 = mixture[160000:320000]  # Second 10 seconds

# Stack into batch [2, 160000]
mixture_batch = np.stack([mixture_1, mixture_2], axis=0)

violon = ['violin', 'violin']
sax = ['saxophone', 'saxophone']
piano = ['piano', 'piano']

with torch.no_grad():
    # Create batch tensor [2, 160000]
    mixture_tensor = torch.from_numpy(mixture_batch).to(device)

    print(f"Mixture batch shape: {mixture_tensor.shape}")
    
    # STFT on batch
    mix_mag, mix_phase = stft.transform(mixture_tensor)  # [2, F, T']
    mix_mag = mix_mag.unsqueeze(1)  # [2, 1, F, T']
    
    print(f"Mix mag shape: {mix_mag.shape}")
    
    # Process all instruments
    mask_violon = model(mix_mag, None, violon)  # [2, 1, F, T']
    mask_sax = model(mix_mag, None, sax)
    mask_piano = model(mix_mag, None, piano)
    
    # Apply masks
    pred_mag_violon = mask_violon * mix_mag  # [2, 1, F, T']
    pred_mag_sax = mask_sax * mix_mag
    pred_mag_piano = mask_piano * mix_mag
    
    # Inverse STFT
    pred_mag_violon_squeezed = pred_mag_violon.squeeze(1)  # [2, F, T']
    pred_wav_violon = stft.inverse(pred_mag_violon_squeezed, mix_phase)  # [2, 160000]
    
    pred_mag_sax_squeezed = pred_mag_sax.squeeze(1)
    pred_wav_sax = stft.inverse(pred_mag_sax_squeezed, mix_phase)
    
    pred_mag_piano_squeezed = pred_mag_piano.squeeze(1)
    pred_wav_piano = stft.inverse(pred_mag_piano_squeezed, mix_phase)
    
    # Convert to numpy
    pred_wav_violon = pred_wav_violon.cpu().numpy()  # [2, 160000]
    pred_wav_sax = pred_wav_sax.cpu().numpy()
    pred_wav_piano = pred_wav_piano.cpu().numpy()
    
    print(f"Predicted violin shape: {pred_wav_violon.shape}")
    print(f"Predicted saxophone shape: {pred_wav_sax.shape}")
    print(f"Predicted piano shape: {pred_wav_piano.shape}")
    
    # Flatten to concatenate both segments: [2, 160000] -> [320000]
    pred_wav_violon_full = pred_wav_violon.flatten()
    pred_wav_sax_full = pred_wav_sax.flatten()
    pred_wav_piano_full = pred_wav_piano.flatten()
    
    print(f"After flatten - violin shape: {pred_wav_violon_full.shape}")

pred_wav_violon_full = normalize_audio(pred_wav_violon_full)
pred_wav_sax_full = normalize_audio(pred_wav_sax_full)
pred_wav_piano_full = normalize_audio(pred_wav_piano_full)

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_path_violon = os.path.join(output_dir, 'pred_violon.wav')
output_path_sax = os.path.join(output_dir, 'pred_sax.wav')
output_path_piano = os.path.join(output_dir, 'pred_piano.wav')

# Use scipy.io.wavfile with proper scaling
from scipy.io import wavfile

# Convert float32 [-1, 1] to int16
def float_to_int16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)

wavfile.write(output_path_violon, 16000, float_to_int16(pred_wav_violon_full))
wavfile.write(output_path_sax, 16000, float_to_int16(pred_wav_sax_full))
wavfile.write(output_path_piano, 16000, float_to_int16(pred_wav_piano_full))

print(f"Predicted violin saved to: {output_path_violon}")
print(f"Predicted saxophone saved to: {output_path_sax}")
print(f"Predicted piano saved to: {output_path_piano}")