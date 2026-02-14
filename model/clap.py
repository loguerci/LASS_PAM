import torch
import torch.nn as nn
import torch.nn.functional as F
import laion_clap
import numpy as np


def _torch_to_numpy_audio(x: torch.Tensor):
    """
    x: (B, T) or (T,)
    returns: np.ndarray float32 in [-1, 1]
    """
    if x.is_cuda:
        x = x.cpu()  # ← CORRECTION: .cpu() au lieu de .cuda()
    x = x.detach()

    return x.numpy().astype(np.float32)


class ClapConditioner(nn.Module):
    def __init__(
        self,
        clap_ckpt: str,
        device: torch.device = torch.device('cuda'),
        use_audio: bool = True,
        use_text: bool = True,
    ):
        super().__init__()

        assert use_audio or use_text, "CLAP needs audio and/or text"

        self.use_audio = use_audio
        self.use_text = use_text
        self.device = device

        # --- Load CLAP ---
        self.clap = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-base"
        )
        self.clap.load_ckpt(clap_ckpt)
        self.clap.to(device)
        self.clap.eval()

        # Freeze CLAP
        for p in self.clap.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def forward(self, audio_ref=None, text=None):
        """
        audio_ref: Tensor [B, T] (mono waveform) - can be on CUDA
        text: List[str] of length B

        returns:
            z_clap: Tensor [B, D_clap] - on same device as audio_ref
        """

        embeddings = []

        if self.use_audio:
            assert audio_ref is not None, "audio_ref is required"
            
            # Convert CUDA tensor to CPU numpy array for CLAP
            audio_np = _torch_to_numpy_audio(audio_ref)
            
            # CLAP processes numpy and returns numpy
            z_audio = self.clap.get_audio_embedding_from_data(audio_np)  # numpy [B, D_clap]
            
            # Convert back to torch tensor on same device as input
            z_audio = torch.from_numpy(z_audio).to(audio_ref.device)
            z_audio = F.normalize(z_audio, dim=-1)
            embeddings.append(z_audio)

        if self.use_text:
            assert text is not None, "text is required"
            
            # CLAP processes text and returns numpy
            z_text = self.clap.get_text_embedding(text)  # numpy [B, D_clap]
            
            # Convert to torch tensor on same device as audio (or self.device)
            device = audio_ref.device if audio_ref is not None else self.device
            z_text = torch.from_numpy(z_text).to(device)
            z_text = F.normalize(z_text, dim=-1)
            embeddings.append(z_text)

        if len(embeddings) == 1:
            z = embeddings[0]
        else:
            z = torch.stack(embeddings, dim=0).mean(dim=0)

        z = F.normalize(z, dim=-1)
        return z
