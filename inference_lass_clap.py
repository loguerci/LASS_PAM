import torch
import torch.nn as nn
from model.LASS_clap import LASS_clap
from utils.stft import STFT
from dataset import LASSClapDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import fast_bss_eval


def get_dataloader(data_dir, shuffle):
    dataset = LASSClapDataset(
        data_dir=data_dir,
        sample_rate=16000,
        segment_samples=int(10 * 16000),
        augment=shuffle,
        cache_in_memory=False
    )
    
    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=shuffle,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    stft = STFT(
        filter_length=1024,
        hop_length=256,
        win_length=1024
    ).to(device)
    
    model = LASS_clap(device=device).to(device)
    
    ckpt_path = '/home/infres/lgosselin-25/LASS_PAM/checkpoints/exp_lr_10e-4_best_0.24/best.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_path = 'data/processed/test'
    test_loader = get_dataloader(test_path, shuffle=False)
    
    total_sdr = 0.0
    total_sar = 0.0
    total_si_sdr = 0.0
    num_samples = 0
    
    pbar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            mixture = batch['mixture'].to(device)       # (B, T)
            target = batch['target'].to(device)         # (B, T)
            reference = batch['reference'].to(device)   # (B, T)
            prompts = batch['prompts']
            
            # STFT
            mix_mag, mix_phase = stft.transform(mixture)      # (B, F, T')
            
            # Add channel dimension
            mix_mag = mix_mag.unsqueeze(1)  # (B, 1, F, T')
            
            # Forward
            mask = model(mix_mag, reference, prompts)  # (B, 1, F, T')
            pred_mag = mask * mix_mag                  # (B, 1, F, T')
            
            # Inverse STFT
            pred_mag_squeezed = pred_mag.squeeze(1)  # (B, F, T')
            pred_wav = stft.inverse(pred_mag_squeezed, mix_phase)  # (B, 1, T) ou (B, T)
            

            if pred_wav.ndim == 3:
                pred_wav = pred_wav.squeeze(1)  # (B, T)
            
            target_wav = target  # (B, T)
            
            # Compute metrics for each sample
            batch_size = pred_wav.shape[0]
            
            for i in range(batch_size):
                pred_i = pred_wav[i:i+1]     # (1, T)
                target_i = target_wav[i:i+1] # (1, T)
                
                sdr, sir, sar = fast_bss_eval.bss_eval_sources(
                        target_i,
                        pred_i,
                        compute_permutation=False
                    )
                    
                si_sdr, _, _= fast_bss_eval.si_bss_eval_sources(
                        target_i,
                        pred_i,
                        compute_permutation=False
                    )
                    
                total_sdr += float(sdr[0])
                total_sir += float(sir[0])
                total_sar += float(sar[0])
                total_si_sdr += float(si_sdr[0])
                num_samples += 1
            
            # Update progress bar
            if num_samples > 0:
                pbar.set_postfix({
                    'SDR': f"{total_sdr / num_samples:.2f}",
                    'SI-SDR': f"{total_si_sdr / num_samples:.2f}",
                    'SIR': f"{total_sir / num_samples:.2f}",
                    'SAR': f"{total_sar / num_samples:.2f}"
                })
    
    print(f"\n{'='*60}")
    print(f"Test Results ({num_samples} samples)")
    print(f"{'='*60}")
    print(f"Average SDR:    {total_sdr / num_samples:.2f} dB")
    print(f"Average SIR:    {total_sir / num_samples:.2f} dB")
    print(f"Average SAR:    {total_sar / num_samples:.2f} dB")
    print(f"Average SI-SDR: {total_si_sdr / num_samples:.2f} dB")
    print(f"{'='*60}")

    #---------------------------------------------------------------------------
    # Process one audio file for qualitative analysis
    #---------------------------------------------------------------------------
    import soundfile as sf
    import os   

    audio_path = 'examples/Dont_know_short.wav'
    mixture, sr = sf.read(audio_path)
    assert sr == 16000, "Sample rate must be 16 kHz"
    mixture_tensor = torch.from_numpy(mixture).float().unsqueeze(0).to(device)  # (1, T)
    mix_mag, mix_phase = stft.transform(mixture_tensor) 
    mix_mag = mix_mag.unsqueeze(1)  # (1, 1, F, T')
    mask_violon = model(mix_mag, None, 'violin')  
    mask_sax = model(mix_mag, None, 'saxophone')
    mask_piano = model(mix_mag, None, 'piano')
    
    pred_mag_violon = mask_violon * mix_mag
    pred_mag_violon_squeezed = pred_mag_violon.squeeze(1)  
    pred_wav_violon = stft.inverse(pred_mag_violon_squeezed, mix_phase)  
    pred_wav_violon = pred_wav_violon.squeeze(1).numpy()  
    
    pred_mag_sax = mask_sax * mix_mag
    pred_mag_sax_squeezed = pred_mag_sax.squeeze(1)  
    pred_wav_sax = stft.inverse(pred_mag_sax_squeezed, mix_phase)  
    pred_wav_sax = pred_wav_sax.squeeze(1).numpy()  
    
    pred_mag_piano = mask_piano * mix_mag
    pred_mag_piano_squeezed = pred_mag_piano.squeeze(1)  
    pred_wav_piano = stft.inverse(pred_mag_piano_squeezed, mix_phase)  
    pred_wav_piano = pred_wav_piano.squeeze(1).numpy()  
    
    output_dir = 'examples/outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path_violon = os.path.join(output_dir, 'pred_violon.wav')
    output_path_sax = os.path.join(output_dir, 'pred_sax.wav')
    output_path_piano = os.path.join(output_dir, 'pred_piano.wav')
    sf.write(output_path_violon, pred_wav_violon, sr)
    sf.write(output_path_sax, pred_wav_sax, sr)
    sf.write(output_path_piano, pred_wav_piano, sr)
    print(f"Predicted violin saved to: {output_path_violon}")
    print(f"Predicted saxophone saved to: {output_path_sax}")
    print(f"Predicted piano saved to: {output_path_piano}")