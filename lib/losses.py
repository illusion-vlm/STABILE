import torch
import torch.nn as nn
import torch.nn.functional as F
    

class Logit_Compensation(nn.Module):
    
    def __init__(self, cls_weight=None):
        super(Logit_Compensation, self).__init__()
        
        self.cls_weight = cls_weight + 1e-12
        self.criterion = nn.BCELoss()
        
    def forward(self, logits, targets):
        
        adjust_logits = logits + self.cls_weight.log()
        loss = self.criterion(torch.sigmoid(adjust_logits), targets)
        
        return loss
    

class UMLLoss(nn.Module):
    
    def __init__(self, temperature=0.1, num_classes=30, cls_weight=None, sample_cls_count=None):
        super(UMLLoss, self).__init__()
        
        self.temperature = temperature
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.per_sample_cls_count = torch.repeat_interleave(sample_cls_count, sample_cls_count, dim=0)
        self.per_sample_cls_weight = torch.repeat_interleave(cls_weight, sample_cls_count, dim=0)
        
        
    def forward(self, feats_anchor, targets_anchor, feats_sample, targets_sample):
        
        repeats = (targets_anchor==1).sum(-1)
        feats_anchor_repeats = torch.repeat_interleave(feats_anchor, repeats, dim=0)
        cls_position = torch.arange(0, self.num_classes).to(feats_anchor.device)
        targets_anchor_repeats = torch.cat([cls_position[target==1] for target in targets_anchor])
        
        mask = torch.cat([(target==targets_sample).unsqueeze(0) for target in targets_anchor_repeats], dim=0).float()
        logits_mask = torch.ones_like(mask).float()
        
        per_sample_cls_weight = self.per_sample_cls_weight.view(1, -1).expand(feats_anchor_repeats.shape[0], feats_sample.shape[0])
        logits = feats_anchor_repeats @ feats_sample.T / self.temperature
        logits = logits + per_sample_cls_weight.log()
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # class-averaging
        per_sample_cls_count = self.per_sample_cls_count.view(1, -1).expand(feats_anchor_repeats.shape[0], feats_sample.shape[0])
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits_sum = exp_logits.div(per_sample_cls_count).sum(dim=1, keepdim=True)
        
        log_prob_repeats = logits - torch.log(exp_logits_sum)
        log_prob_repeats = (mask * log_prob_repeats).sum(1) / mask.sum(1)
        log_prob_batches = log_prob_repeats.split(repeats.tolist(), dim=0)
        mean_log_prob_per_batch = torch.cat([m.sum(0).unsqueeze(0) for m in log_prob_batches], dim=0) / repeats
        loss = - mean_log_prob_per_batch
        loss = loss.mean()
            
        return loss

    
if __name__ == '__main__':
    
    feats_anchor = torch.randn([3,5])
    # targets_anchor = torch.tensor([[0,2,-1],[1,2,-1],[1,-1,-1]])
    targets_anchor = torch.tensor([[1,0,1],[0,1,1],[0,1,0]])
    feats_sample = torch.randn([10,5])
    targets_sample = torch.tensor([0,0,1,1,1,2,2,2,2,2])
    cls_weight = torch.tensor([0.2,0.3,0.5])
    sample_cls_count = torch.tensor([2,3,5])
    uml = UMLLoss(0.07, 3, cls_weight, sample_cls_count)
    uml(feats_anchor, targets_anchor, feats_sample, targets_sample)
    
    
