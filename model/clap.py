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
        x = x.cuda()
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

        # --- Load CLAP ---
        self.clap = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-base"
        )
        self.clap.load_ckpt(clap_ckpt)
        self.clap.to('cuda')
        self.clap.eval()

        # Freeze CLAP
        for p in self.clap.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def forward(self, audio_ref=None, text=None):
        """
        audio_ref: Tensor [B, T] (mono waveform)
        text: List[str] of length B

        returns:
            z_clap: Tensor [B, D_clap]
        """

        embeddings = []

        if self.use_audio:
            assert audio_ref is not None, "audio_ref is required"
            audio_np = _torch_to_numpy_audio(audio_ref)
            z_audio = self.clap.get_audio_embedding_from_data(audio_np)  # [B, D_clap]
            z_audio = torch.from_numpy(z_audio).to('cuda')
            z_audio = F.normalize(z_audio, dim=-1)
            embeddings.append(z_audio)

        if self.use_text:
            assert text is not None, "text is required"
            z_text = self.clap.get_text_embedding(text)  # [B, D_clap]
            z_text = torch.from_numpy(z_text).to('cuda')
            z_text = F.normalize(z_text, dim=-1)
            embeddings.append(z_text)

        if len(embeddings) == 1:
            z = embeddings[0]
        else:
            z = torch.stack(embeddings, dim=0).mean(dim=0)

        z = F.normalize(z, dim=-1)
        return z