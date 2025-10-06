import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from copy import deepcopy

from config import Config

class SimpleProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden = (input_dim + output_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class CrossModalAttentionEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.p_dim, n_heads = config.projection_dim, config.attention_heads
        self.video_ex = SimpleProjector(config.video_dim, self.p_dim)
        self.speech_ex = SimpleProjector(config.speech_dim, self.p_dim)
        self.text_ex = SimpleProjector(config.text_dim, self.p_dim)
        self.atts = nn.ModuleList([nn.MultiheadAttention(self.p_dim, n_heads, batch_first=True) for _ in range(6)])

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.p_dim,
            nhead=n_heads,
            dim_feedforward=self.p_dim * 4,
            batch_first=True,
            activation='gelu'
        )
        self.fusion_attention = nn.TransformerEncoder(fusion_layer, num_layers=1)
        self.final_mlp = nn.Linear(self.p_dim, self.p_dim)

    def forward(self, v, s, t):
        v_f, s_f, t_f = self.video_ex(v), self.speech_ex(s), self.text_ex(t)

        v_q, s_q, t_q = v_f.unsqueeze(1), s_f.unsqueeze(1), t_f.unsqueeze(1)
        v_s, _ = self.atts[0](v_q, s_q, s_q)
        v_t, _ = self.atts[1](v_q, t_q, t_q)
        s_v, _ = self.atts[2](s_q, v_q, v_q)
        s_t, _ = self.atts[3](s_q, t_q, t_q)
        t_v, _ = self.atts[4](t_q, v_q, v_q)
        t_s, _ = self.atts[5](t_q, s_q, s_q)

        feature_sequence = torch.stack([
            v_f, s_f, t_f,
            v_s.squeeze(1), v_t.squeeze(1),
            s_v.squeeze(1), s_t.squeeze(1),
            t_v.squeeze(1), t_s.squeeze(1)
        ], dim=1)

        fused_sequence = self.fusion_attention(feature_sequence)
        fused_representation = fused_sequence.mean(dim=1)
        final_output = self.final_mlp(fused_representation)
        return final_output

class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        hidden = (latent_dim + output_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class MultitaskUnsupervisedModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.K, self.m, self.T = config.moco_k, config.moco_m, config.temperature

        # Encoders
        self.base_encoder_q = CrossModalAttentionEncoder(config)
        self.base_encoder_k = CrossModalAttentionEncoder(config)

        # Projection heads
        self.projection_head_q = nn.Sequential(
            nn.Linear(config.projection_dim, config.projection_dim), nn.ReLU(),
            nn.Linear(config.projection_dim, config.moco_dim)
        )
        self.projection_head_k = nn.Sequential(
            nn.Linear(config.projection_dim, config.projection_dim), nn.ReLU(),
            nn.Linear(config.projection_dim, config.moco_dim)
        )

        # Decoders
        self.video_decoder  = ModalityDecoder(config.projection_dim, config.video_dim)
        self.speech_decoder = ModalityDecoder(config.projection_dim, config.speech_dim)
        self.text_decoder   = ModalityDecoder(config.projection_dim, config.text_dim)

        # Initialize key encoder from query encoder
        for param_q, param_k in zip(self.base_encoder_q.parameters(), self.base_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo queue
        self.register_buffer("queue", torch.randn(config.moco_dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.base_encoder_q.parameters(), self.base_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward_contrastive(self, im_q, im_k):
        v_q, s_q, t_q = im_q
        v_k, s_k, t_k = im_k

        rep_q = self.base_encoder_q(v_q, s_q, t_q)
        q = self.projection_head_q(rep_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            rep_k = self.base_encoder_k(v_k, s_k, t_k)
            k = self.projection_head_k(rep_k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

    def forward_reconstruction(self, v_aug, s_aug, t_aug):
        fused_rep = self.base_encoder_q(v_aug, s_aug, t_aug)
        v_recon = self.video_decoder(fused_rep)
        s_recon = self.speech_decoder(fused_rep)
        t_recon = self.text_decoder(fused_rep)
        return v_recon, s_recon, t_recon

class SupervisedClassifier(nn.Module):
    def __init__(self, base_encoder, config: Config):
        super().__init__()
        self.base_encoder = base_encoder
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        self.classifier_head = nn.Sequential(
            nn.Linear(config.projection_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, config.n_classes)
        )

    def forward(self, v, s, t):
        with torch.no_grad():
            fused_representation = self.base_encoder(v, s, t)
        return self.classifier_head(fused_representation)
