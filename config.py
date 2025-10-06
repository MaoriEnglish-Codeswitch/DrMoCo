import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    data_dir = 'data_folder'
    video_dim = 1024
    speech_dim = 512
    text_dim = 1024

    projection_dim = 512
    attention_heads = 8

    contrastive_weight = 1.0
    recon_mask_weight = 1.0
    recon_clean_weight = 1.2


    adaptive_loss_scaling = True
    als_beta = 0.9
    als_tau = 0.4
    als_warmup_epochs = 10
    als_min_scale = 0.5
    als_max_scale = 2.0

    moco_dim = 1024
    moco_k = 4096
    moco_m = 0.999

    temperature = 0.06
    mixup_alpha = 0.5

    feature_mask_ratio = 0.2

    unsupervised_epochs = 80
    unsupervised_lr = 1e-4

    supervised_epochs = 10
    supervised_lr = 1e-3

    weight_decay = 1e-5

    n_classes = 2
    batch_size = 128
