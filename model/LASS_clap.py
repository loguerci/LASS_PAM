import torch
import torch.nn as nn
import torch.nn.functional as F
from .resunet_film import UNetRes_FiLM
from .clap import ClapConditioner

class LASS_clap(nn.Module):
    def __init__(self, device='cpu'):
        super(LASS_clap, self).__init__()
        # Plus besoin de Text_Encoder puisque CLAP le remplace
        # self.text_embedder = Text_Encoder(device)  # SUPPRIMER
        
        self.UNet = UNetRes_FiLM(channels=1, cond_embedding_dim=256)
        self.clap_conditioner = ClapConditioner(
            clap_ckpt='/home/lolo/ATIAM/PAM/LASS_PAM/pretrained/music_audioset_epoch_15_esc_90.14.pt',
            device=device,
            use_audio=True,
            use_text=True
        )

        self.proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.device = device

    def forward(self, x, ref, caption):
        """
        Args:
            x: input mixture spectrogram (Batch, 1, T, F)
            ref: reference audio for each source (Batch, audio_samples) 
            caption: list of text descriptions (e.g., ["flute", "violin"])
        Returns:
            mask: predicted mask (Batch, 1, T, F)
        """
        # Get CLAP conditioning
        cond_vec = self.clap_conditioner(
            audio_ref=ref,
            text=caption
        )  # [B, 512]
        cond_vec = self.proj(cond_vec)  # [B, 256]
        
        # Apply U-Net with FiLM conditioning
        print(cond_vec.shape, x.shape)
        mask = self.UNet(x, cond_vec, cond_vec)  # encoder and decoder conditioning
        mask = torch.sigmoid(mask)
        
        return mask
    
    def load_pretrained_unet(self, ckpt_path, strict=False):
        """
        Load pretrained LASS weights, keeping only compatible UNet parameters.
        
        Args:
            ckpt_path: path to LASS checkpoint
            strict: if True, all keys must match (will fail). If False, load what matches.
        """
        print(f"Loading pretrained weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        # Filter only UNet parameters (ignore text_embedder, etc.)
        unet_state_dict = OrderedDict()
        for k, v in new_state_dict.items():
            # Keep only parameters that start with expected prefixes
            if k.startswith('UNet') or k.startswith('ss') or k.startswith('query_net'):
                # Map old UNet names to new ones if needed
                new_key = k.replace('ss.', 'UNet.').replace('query_net.', 'UNet.')
                unet_state_dict[new_key] = v
        
        # Load with strict=False to allow missing keys
        incompatible = self.load_state_dict(unet_state_dict, strict=False)
        
        print(f"✓ Loaded {len(unet_state_dict)} parameters")
        if incompatible.missing_keys:
            print(f"⚠ Missing keys (will be randomly initialized): {len(incompatible.missing_keys)}")
            #print(f"  Examples: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"⚠ Unexpected keys (ignored): {len(incompatible.unexpected_keys)}")
        
        return incompatible


if __name__ == "__main__": # run with "python -m model.LASS_clap"
    
    device = torch.device('cpu')
    ckpt_path = '/home/lolo/ATIAM/PAM/LASS_PAM/pretrained/LASSNet.pt'
    
    # 1. Initialize model
    print("\n1. Initializing LASS_clap...")
    model = LASS_clap(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 2. Try to load pretrained weights
    print("\n2. Loading pretrained LASS weights...")
    try:
        incompatible = model.load_pretrained_unet(ckpt_path, strict=False)
        print("   ✓ Pretrained weights loaded (partially)")
    except Exception as e:
        print(f"   ✗ Could not load pretrained weights: {e}")
        print("   → Model will use random initialization")
    
    # 3. Test forward pass
    print("\n3. Testing forward pass...")
    model.eval()
    x = torch.randn(1, 1, 16384, 513)
    ref = torch.randn(1, 16000*5)
    caption = ["flute"]
    print(x.shape, ref.shape, len(caption))
    with torch.no_grad():
        mask = model(x, ref, caption)
    print(mask.shape)  # should be (1, 1, 16384, 513)