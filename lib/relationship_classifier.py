import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(*channels):
    num_layers = len(channels) - 2
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(channels[-2], channels[-1]))
    return nn.Sequential(*layers)


class RelationshipClassifier(nn.Module):
    
    def __init__(self, embed_dim=None, dropout=None, num_classes=26, K=None, contrastive=True):
        
        super(RelationshipClassifier, self).__init__()
        
        self.n_cls = num_classes
        self.K = K
        self.contrastive = contrastive
        
        self.encoder = nn.MultiheadAttention(embed_dim, 1, dropout=dropout)
        self.fc_out = nn.Linear(embed_dim, num_classes)

        if contrastive and self.K:
            self.fc_mlp = MLP(embed_dim, embed_dim, 1024)

            # create the queue
            self.register_buffer("queue_k", F.normalize(torch.randn(K, 1024), dim=1))
            self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))

            self.register_buffer("queue_i", torch.arange(-K, 0))
        
            # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))


    def set_cls_weight(self, cls_weight, base_K=5):

        device = cls_weight.device
        cls_position = torch.arange(1, self.n_cls + 1, device=device) * base_K \
            + (self.K - self.n_cls * base_K) * cls_weight.cumsum(0)
        end_idx = cls_position.ceil().long()
        start_idx = end_idx.clone()
        start_idx[1:] = start_idx[:-1].clone()
        start_idx[0] = 0
        self.base_k = base_K
        self.cls_start_idx = start_idx
        self.K_per_cls = end_idx - start_idx
        for c, (s, e) in enumerate(zip(start_idx, end_idx)):
            self.queue_l[s:e] = c

        self.buffer_freq_correction = cls_weight / self.K_per_cls * self.K
        print(self.cls_start_idx)
        print(self.K_per_cls)

        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, in_idx):
        
        keys = keys.clone().detach()
        labels = labels.clone().detach()
        in_idx = in_idx.clone().detach()
        
        intra_cls_idx = F.one_hot(labels, self.n_cls).cumsum(0) - 1
        intra_cls_idx = intra_cls_idx.gather(1, labels.unsqueeze(1)).squeeze(1)
        max_K_per_idx = self.K_per_cls.gather(0, labels)
        mask = intra_cls_idx < max_K_per_idx
        
        keys = keys[mask]
        labels = labels[mask]
        in_idx = in_idx[mask]
        intra_cls_idx = intra_cls_idx[mask]
        max_K_per_idx = max_K_per_idx[mask]
        
        cls_start_idx = self.cls_start_idx.gather(0, labels)
        offset = (self.queue_ptr.gather(0, labels) + intra_cls_idx) % max_K_per_idx
        
        target_pos = cls_start_idx + offset
        self.queue_k.scatter_(0, target_pos.unsqueeze(1).repeat(1, keys.shape[1]), keys.detach())
        self.queue_l.scatter_(0, target_pos, labels)
        self.queue_i.scatter_(0, target_pos, in_idx)
        
        samples_per_cls = labels.bincount(minlength=self.n_cls)
        self.queue_ptr = (self.queue_ptr + samples_per_cls) % self.K_per_cls
            
        
    def forward(self, feats, bs, labels=None, in_idx=None):

        feats = feats.unsqueeze(1)
        feats, _ = self.encoder(feats, feats, feats)
        feats = feats.squeeze(1)
        
        feats_cls = feats[:bs]
        logits = self.fc_out(feats_cls)

        if self.training and self.contrastive:

            _, feats_anchor, feats_contras= torch.split(feats, [bs, bs, bs], dim=0)

            feats_anchor = F.normalize(self.fc_mlp(feats_anchor), dim=1)
            feats_contras = F.normalize(self.fc_mlp(feats_contras), dim=1)
            
            self._dequeue_and_enqueue(feats_contras, labels, in_idx)

            feats_sample = self.queue_k.clone().detach()
            targets_sample = self.queue_l.clone().detach()
            idx_sample = self.queue_i.clone().detach()
            
            return logits, feats_anchor, feats_sample, targets_sample, idx_sample
        else:
            return logits, feats_cls
        