import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import Text_Encoder
from .resunet_film import UNetRes_FiLM
from .clap import ClapConditioner

class LASS_clap(nn.Module):
    def __init__(self, device='cuda'):
        super(LASS_clap, self).__init__()
        self.text_embedder = Text_Encoder(device)
        self.UNet = UNetRes_FiLM(channels=1, cond_embedding_dim=256)
        self.clap_conditioner = ClapConditioner(
            clap_ckpt='/home/lolo/ATIAM/PAM/LASS_PAM/pretrained/music_audioset_epoch_15_esc_90.14.pt',
            device=device,
            use_audio=False,
            use_text=True,
            out_dim=256
        )

    def forward(self, x, ref, caption):
        # x: (Batch, 1, T, 128))
        
        cond_vec = self.clap_conditioner(
            audio_ref=ref,
            text=caption
        )  # [B, D_clap]
        dec_cond_vec = cond_vec

        mask = self.UNet(x, cond_vec, dec_cond_vec)
        mask = torch.sigmoid(mask)
        return mask
