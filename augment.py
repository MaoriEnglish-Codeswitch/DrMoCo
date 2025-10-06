import torch, random
import numpy as np

def feature_mask_aug(v, s, t, config):
    v_aug, s_aug, t_aug = v.clone(), s.clone(), t.clone()
    modality_to_mask = random.randint(0, 2)
    mask_indices = None

    if modality_to_mask == 0:
        num_dims = v.shape[-1]
        num_mask = int(num_dims * config.feature_mask_ratio)
        mask_indices = torch.randperm(num_dims, device=v.device)[:num_mask]
        v_aug[:, mask_indices] = 0.0

    elif modality_to_mask == 1:
        num_dims = s.shape[-1]
        num_mask = int(num_dims * config.feature_mask_ratio)
        mask_indices = torch.randperm(num_dims, device=s.device)[:num_mask]
        s_aug[:, mask_indices] = 0.0

    else:
        num_dims = t.shape[-1]
        num_mask = int(num_dims * config.feature_mask_ratio)
        mask_indices = torch.randperm(num_dims, device=t.device)[:num_mask]
        t_aug[:, mask_indices] = 0.0

    return (v_aug, s_aug, t_aug), modality_to_mask, mask_indices

def apply_gaussian_noise(v, s, t, noise_factor=0.1):
    v_noise = v + torch.randn_like(v) * noise_factor
    s_noise = s + torch.randn_like(s) * noise_factor
    t_noise = t + torch.randn_like(t) * noise_factor
    return v_noise, s_noise, t_noise

def apply_mixup_to_batch(batch_features, config):
    v_batch, s_batch, t_batch = batch_features
    batch_size = v_batch.size(0)
    indices = torch.randperm(batch_size)
    indices = indices.to(v_batch.device, non_blocking=True)
    lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
    v_mixed = lam * v_batch + (1 - lam) * v_batch[indices]
    s_mixed = lam * s_batch + (1 - lam) * s_batch[indices]
    t_mixed = lam * t_batch + (1 - lam) * t_batch[indices]
    return (v_mixed, s_mixed, t_mixed), lam, indices
