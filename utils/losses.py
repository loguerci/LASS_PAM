"""
Loss functions and metrics for LASS_clap
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """
    Multi-scale spectral loss
    Similar to LASS paper
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], alpha=1.0, beta=1.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.alpha = alpha  # Weight for magnitude loss
        self.beta = beta    # Weight for log-magnitude loss
    
    def forward(self, pred_spec, target_spec):
        """
        Args:
            pred_spec: (B, 1, F, T) predicted spectrogram
            target_spec: (B, 1, F, T) target spectrogram
        """
        # Simple L1 loss on log-mel spectrograms
        # (les spectrogrammes sont déjà en log dans le dataset)
        loss = F.l1_loss(pred_spec, target_spec)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: L1 + Spectral
    """
    def __init__(self, l1_weight=1.0, spectral_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.spectral_loss = SpectralLoss()
    
    def forward(self, pred_spec, target_spec):
        # L1 loss
        l1_loss = F.l1_loss(pred_spec, target_spec)
        
        # Spectral loss (déjà inclus dans L1 pour log-mel spec)
        # Mais on peut ajouter des pertes multi-échelles si besoin
        
        total_loss = self.l1_weight * l1_loss
        
        return total_loss, {'l1_loss': l1_loss.item()}


def get_loss_function(loss_type="l1"):
    """Factory function for loss"""
    if loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "spectral":
        return SpectralLoss()
    elif loss_type == "combined":
        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Metrics
def compute_sdr(pred, target, epsilon=1e-8):
    """
    Signal-to-Distortion Ratio (SDR)
    
    Args:
        pred: (B, C, F, T) predicted spectrogram
        target: (B, C, F, T) target spectrogram
    
    Returns:
        sdr: scalar
    """
    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # SDR
    signal_power = torch.sum(target_flat ** 2)
    noise_power = torch.sum((pred_flat - target_flat) ** 2)
    
    sdr = 10 * torch.log10(signal_power / (noise_power + epsilon) + epsilon)
    
    return sdr.item()


def compute_metrics(pred_spec, target_spec):
    """
    Compute all metrics
    
    Returns:
        dict of metrics
    """
    metrics = {}
    
    # L1 loss
    metrics['l1'] = F.l1_loss(pred_spec, target_spec).item()
    
    # MSE loss
    metrics['mse'] = F.mse_loss(pred_spec, target_spec).item()
    
    # SDR
    metrics['sdr'] = compute_sdr(pred_spec, target_spec)
    
    return metrics