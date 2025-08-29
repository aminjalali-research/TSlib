import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, tau=1.0, 
                                amc_instance=None, amc_temporal=None, amc_margin=0.5,
                                adaptive_temp=False, temp_alpha=0.5, temp_A0_inst=0.1, temp_A0_temp=0.2):
    """
    Integrated hierarchical contrastive loss combining features from both versions.
    
    Args:
        z1, z2: Input tensors
        alpha: Balance between instance and temporal loss
        temporal_unit: Minimum unit for temporal contrast
        tau: Base temperature parameter
        amc_instance: AMC coefficient for instance loss (None to disable)
        amc_temporal: AMC coefficient for temporal loss (None to disable)
        amc_margin: Margin for AMC loss
        adaptive_temp: Whether to use adaptive temperature (from losses2.py)
        temp_alpha: Alpha parameter for adaptive temperature
        temp_A0_inst: A0 parameter for instance adaptive temperature
        temp_A0_temp: A0 parameter for temporal adaptive temperature
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(
                z1, z2, tau=tau, amc_coef=amc_instance, amc_margin=amc_margin,
                adaptive_temp=adaptive_temp, temp_alpha=temp_alpha, temp_A0=temp_A0_inst
            )
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(
                    z1, z2, tau=tau, amc_coef=amc_temporal, amc_margin=amc_margin,
                    adaptive_temp=adaptive_temp, temp_alpha=temp_alpha, temp_A0=temp_A0_temp
                )
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(
                z1, z2, tau=tau, amc_coef=amc_temporal, amc_margin=amc_margin,
                adaptive_temp=adaptive_temp, temp_alpha=temp_alpha, temp_A0=temp_A0_inst
            )
        d += 1
    
    return loss / d

def instance_contrastive_loss(z1, z2, tau=1.0, amc_coef=0, amc_margin=0.5,
                            adaptive_temp=False, temp_alpha=0.5, temp_A0=0.1):
    """
    Instance contrastive loss with optional adaptive temperature.
    
    Args:
        z1, z2: Input tensors (BxTxC)
        tau: Base temperature parameter
        amc_coef: AMC coefficient (0 to disable, positive value to enable)
        amc_margin: Margin for AMC loss
        adaptive_temp: Whether to use adaptive temperature from losses2.py
        temp_alpha: Alpha parameter for adaptive temperature
        temp_A0: A0 parameter for adaptive temperature
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]     # Remove diagonal
    
    i = torch.arange(B, device=z1.device)
    
    if adaptive_temp:
        # Adaptive temperature from losses2.py
        A = ((logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2).detach()
        tau_adaptive = tau * (1. + temp_alpha * (A - temp_A0))
        logits = -F.log_softmax(logits, dim=-1) / tau_adaptive
    else:
        # Fixed temperature from losses.py (more efficient)
        logits = torch.div(logits, tau)
        logits = -F.log_softmax(logits, dim=-1)
    
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    
    if amc_coef and amc_coef > 0:
        # Use optimized vectorized AMC from losses.py
        amc_loss = amc3d_vectorized('cuda', z, amc_margin=amc_margin)
        return loss + amc_coef * amc_loss
    else:
        return loss

def temporal_contrastive_loss(z1, z2, tau=1.0, amc_coef=0, amc_margin=0.5,
                            adaptive_temp=False, temp_alpha=0.5, temp_A0=0.2):
    """
    Temporal contrastive loss with optional adaptive temperature.
    
    Args:
        z1, z2: Input tensors (BxTxC)  
        tau: Base temperature parameter
        amc_coef: AMC coefficient (0 to disable, positive value to enable)
        amc_margin: Margin for AMC loss
        adaptive_temp: Whether to use adaptive temperature from losses2.py
        temp_alpha: Alpha parameter for adaptive temperature
        temp_A0: A0 parameter for adaptive temperature
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    
    t = torch.arange(T, device=z1.device)
    
    if adaptive_temp:
        # Adaptive temperature from losses2.py
        A = ((logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2).detach()
        tau_adaptive = tau * (1. + temp_alpha * (A - temp_A0))
        logits = -F.log_softmax(logits, dim=-1) / tau_adaptive
    else:
        # Fixed temperature from losses.py (more efficient) - CORRECTED ORDER
        logits = torch.div(logits, tau)
        logits = -F.log_softmax(logits, dim=-1)
    
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
    if amc_coef and amc_coef > 0:
        # Use optimized vectorized AMC from losses.py
        amc_loss = amc3d_vectorized('cuda', z, amc_margin=amc_margin)
        return loss + amc_coef * amc_loss
    else:
        return loss

# Optimized vectorized AMC implementation from losses.py
def amc3d_vectorized(device, features, amc_margin=0.5):
    """
    Optimized vectorized Angular Margin Contrastive loss.
    More efficient than the loop-based version in losses2.py.
    """
    features = F.normalize(features, dim=-1)
    similarity_matrix = torch.matmul(features, features.transpose(-1, -2))
    bs = features.shape[1] // 2
    labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    labels = labels.repeat(features.shape[0], 1, 1)
    
    mask = torch.eye(labels.shape[1], dtype=torch.bool).to(device)
    mask = mask.repeat(features.shape[0], 1, 1)
    labels = labels[~mask].view(features.shape[0], labels.shape[1], -1)
    similarity_matrix = similarity_matrix[~mask].view(features.shape[0], similarity_matrix.shape[1], -1)
    
    positives = similarity_matrix[labels.bool()].view(features.shape[0], labels.shape[1], -1)
    negatives = similarity_matrix[~labels.bool()].view(features.shape[0], labels.shape[1], -1)
    
    negatives = torch.clamp(negatives, min=-1+1e-10, max=1-1e-10)
    clip = torch.acos(negatives)
    b1 = amc_margin - clip
    mask = b1 > 0
    l1 = torch.sum((mask * b1) ** 2, dim=[1, 2])
    
    positives = torch.clamp(positives, min=-1+1e-10, max=1-1e-10)
    l2 = torch.acos(positives)
    l2 = torch.sum(l2 ** 2, dim=[1, 2])
    
    loss = (l1 + l2) / 25
    total_loss = torch.sum(loss)
    return total_loss

# Loop-based AMC implementation from losses2.py (kept for compatibility)
def amc3d_loop(device, features, amc_margin=0.5):
    """
    Loop-based AMC implementation from losses2.py.
    Less efficient but kept for compatibility.
    """
    total_loss = torch.tensor(0.0).to(device)
    main_features = features
    
    for i in range(len(main_features)):
        features = main_features[i]
        bs = features.shape[0] // 2

        labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Select positives and negatives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        negatives = torch.clamp(negatives, min=-1+1e-10, max=1-1e-10)
        clip = torch.acos(negatives)
        b1 = amc_margin - clip
        mask = b1 > 0
        l1 = torch.sum((mask * b1) ** 2)
        
        positives = torch.clamp(positives, min=-1+1e-10, max=1-1e-10)
        l2 = torch.acos(positives)
        l2 = torch.sum(l2 ** 2)
        
        loss = (l1 + l2) / 25
        total_loss = total_loss + loss

    return total_loss

# Simple AMC for flattened features (from losses2.py)
def amc_flattened(device, features, amc_margin=0.5):
    """
    AMC loss for flattened features (used in total_loss from losses2.py).
    """
    bs = features.shape[0] // 2
    labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # Discard the main diagonal
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives and negatives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    negatives = torch.clamp(negatives, min=-1+1e-10, max=1-1e-10)
    clip = torch.acos(negatives)
    b1 = amc_margin - clip
    mask = b1 > 0
    l1 = torch.sum((mask * b1) ** 2)
    
    positives = torch.clamp(positives, min=-1+1e-10, max=1-1e-10)
    l2 = torch.acos(positives)
    l2 = torch.sum(l2 ** 2)
    
    loss = (l1 + l2) / 25
    return loss

def total_loss(z1, z2, alpha=0.5, temporal_unit=0, tau=1.0, 
               amc_instance=None, amc_temporal=None, amc_margin=0.5,
               amc_flattened_coef=0.1, adaptive_temp=False,
               temp_alpha=0.5, temp_A0_inst=0.1, temp_A0_temp=0.2):
    """
    Total loss function combining hierarchical loss and optional flattened AMC loss.
    This is the extended version from losses2.py with all options.
    
    Args:
        amc_flattened_coef: Coefficient for additional AMC loss on flattened features
        Other args: Same as hierarchical_contrastive_loss
    """
    # Hierarchical contrastive loss
    hier_loss = hierarchical_contrastive_loss(
        z1, z2, alpha=alpha, temporal_unit=temporal_unit, tau=tau,
        amc_instance=amc_instance, amc_temporal=amc_temporal, amc_margin=amc_margin,
        adaptive_temp=adaptive_temp, temp_alpha=temp_alpha, 
        temp_A0_inst=temp_A0_inst, temp_A0_temp=temp_A0_temp
    )
    
    # Optional additional AMC loss on flattened features (from losses2.py)
    if amc_flattened_coef and amc_flattened_coef > 0:
        z1_flat = z1.reshape(len(z1), -1)
        z2_flat = z2.reshape(len(z2), -1)
        features_flat = torch.cat((z1_flat, z2_flat), dim=0)
        amc_loss = amc_flattened(device='cuda', features=features_flat, amc_margin=amc_margin)
        return hier_loss + amc_flattened_coef * amc_loss
    else:
        return hier_loss

# Convenience functions for easy migration from both files
def losses_v1_compatible(z1, z2, alpha=0.5, temporal_unit=0, tau=1.0, 
                        amc_instance=None, amc_temporal=None, amc_margin=0.5):
    """
    Wrapper for losses.py compatibility (optimized version).
    Uses fixed temperature and vectorized AMC.
    """
    return hierarchical_contrastive_loss(
        z1, z2, alpha=alpha, temporal_unit=temporal_unit, tau=tau,
        amc_instance=amc_instance, amc_temporal=amc_temporal, amc_margin=amc_margin,
        adaptive_temp=False  # Use fixed temperature for efficiency
    )

def losses_v2_compatible(z1, z2, alpha=0.5, temporal_unit=0, tau=1.0, 
                        amc_instance=1, amc_temporal=3):
    """
    Wrapper for losses2.py compatibility (experimental version).
    Uses adaptive temperature and includes flattened AMC loss.
    """
    return total_loss(
        z1, z2, alpha=alpha, temporal_unit=temporal_unit, tau=tau,
        amc_instance=amc_instance, amc_temporal=amc_temporal, amc_margin=0.5,
        amc_flattened_coef=0.1, adaptive_temp=True,
        temp_alpha=0.5, temp_A0_inst=0.1, temp_A0_temp=0.2
    )
