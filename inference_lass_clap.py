import torch
import torch.nn as nn
from model.LASS_clap import LASS_clap
from utils.stft import STFT
from dataset import LASSClapDataset, collate_fn
from torch.utils.data import DataLoader
import tqdm
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
        batch_size=16,
        shuffle=shuffle,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

if __name__ == '__main__':

    device = 'cuda'
    stft = STFT()
    model = nn.DataParallel(LASS_clap(device)).to(device)
    ckpt_path = ''
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    val_loader = get_dataloader(None, shuffle=False) 

    pbar = tqdm(val_loader, desc=f"Validation")
    total_sdr, total_sir, total_sar, total_mae = 0, 0, 0, 0

    for batch in pbar:
        mixture = batch['mixture'].to(device)
        target = batch['target'].to(device)
        reference = batch['reference'].to(device)
        prompts = batch['prompts']
            
        # STFT
        mix_mag, _ = stft.transform(mixture)
        target_mag, _ = stft.transform(target)
            
        # Forward
        mask = model(mix_mag, reference, prompts)
        pred = mask * mix_mag

        # Metrics 
        pred_wav = stft.inverse(pred.cuda().detach(), _)
        target_wav = stft.inverse(target_mag.cuda().detach(), _)

        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(pred_wav, target_wav)
        total_sdr += sdr
        total_sir += sir
        total_sar += sar

    num_batches = len(val_loader)
    print(f"Average SDR: {total_sdr / num_batches:.2f} dB")
    print(f"Average SIR: {total_sir / num_batches:.2f} dB")
    print(f"Average SAR: {total_sar / num_batches:.2f} dB")

