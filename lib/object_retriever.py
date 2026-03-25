import copy
import torch
from torch import nn
import torch.nn.functional as F

from lib.stanet import STANet


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.multi_head = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, entry_q, entry_k, entry_v, input_key_padding_mask):
        # encoder
        src, encoder_weights = self.multi_head(entry_q, entry_k, entry_v, key_padding_mask=input_key_padding_mask)

        src = entry_q + self.dropout1(src)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, encoder_weights
    

class SelfAttention(nn.Module):
    
    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1, num_layers=1):
        super().__init__()
        encoder_layer = AttentionLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, _input, input_key_padding_mask):
        _output = _input
        weights = torch.zeros([self.num_layers, _output.shape[1], _output.shape[0], _output.shape[0]]).to(_output.device)

        for i, layer in enumerate(self.layers):
            _output, encoder_weights = layer(_output, _output, _output, input_key_padding_mask)
            weights[i] = encoder_weights

        return _output


class CrossAttention(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1, num_layers=1):
        super().__init__()
        encoder_layer = AttentionLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input_q, input_k, input_v, input_key_padding_mask):
        output_q = input_q
        output_k = input_k
        output_v = input_v
        weights = torch.zeros([self.num_layers, output_q.shape[1], output_q.shape[0], output_q.shape[0]]).to(output_q.device)

        for i, layer in enumerate(self.layers):
            output_v, encoder_weights = layer(output_q, output_k, output_v, input_key_padding_mask)
            output_k = output_v
            weights[i] = encoder_weights
        
        return output_v


class ObjectRetriever(nn.Module):
    
    def __init__(self, img_dim=None,
                 obj_dim=None,
                 att_nhead=None,
                 att_dim_feedforward=None,
                 att_dropout=None,
                 att_layer_num=None,
                 sta_obj_classes_num=None):
        
        super(ObjectRetriever, self).__init__()

        self.stanet = STANet(sta_obj_classes_num)
        self.i2o_fc = nn.Linear(img_dim, obj_dim)
        # self.cross_attention = CrossAttention(embed_dim=obj_dim, nhead=att_nhead, dim_feedforward=att_dim_feedforward, dropout=att_dropout, num_layers=att_layer_num)
        self.cross_attention = nn.MultiheadAttention(obj_dim, att_nhead, dropout=att_dropout)

        self.self_attention = SelfAttention(embed_dim=obj_dim, nhead=att_nhead, dim_feedforward=att_dim_feedforward, dropout=att_dropout, num_layers=att_layer_num)
        self.position_embedding = nn.Embedding(3, obj_dim) #previous and present and next frame
        
    
    def forward(self, entry):
        
        img_feat_base = entry['base_features']
        img_feat_top = entry['top_features']
        # aud_feat = entry['audio_features']
        
        obj_idx = entry['boxes'][:, 0].long()
        obj_features = entry['obj_features']
            
        # img_features, _xss, _xsss, _map = self.stanet(img_feat_top, aud_feat)
        img_features, _xss, _xsss, _map = self.stanet(img_feat_top)
        entry['img_distribution_1'] = _xss
        entry['img_distribution_2'] = _xsss
        entry['saliency_map'] = _map
        
        img_features = self.i2o_fc(img_features[obj_idx])
        
        img_input = img_features.unsqueeze(1)
        obj_input = obj_features.unsqueeze(1)
        cross_obj_output, _ = self.cross_attention(img_input, obj_input, obj_input)
        cross_obj_output = cross_obj_output.squeeze(1)
        obj_features = 0.5 * obj_features + 0.5 * cross_obj_output

        b = int(obj_idx[-1] + 1)
        # l = torch.sum(obj_idx == torch.mode(obj_idx)[0])
        
        assert b != 0, 'No object detected in current video clip!!!'
        
        if b > 2:
            l = 0
            for j in range(b - 2):
                length = torch.sum((obj_idx == j) + (obj_idx == j+1) + (obj_idx == j+2))
                if length > l:
                    l = length
        
            obj_input = torch.zeros([l, b-2, obj_features.shape[1]]).to(obj_features.device)
            position_embed = torch.zeros([l, b-2, obj_features.shape[1]]).to(obj_features.device)
            idx = -torch.ones([l, b-2]).to(obj_features.device)
            
            # padding, frame_size = 3
            for j in range(b - 2):
                obj_input[:torch.sum((obj_idx == j) + (obj_idx == j + 1) + (obj_idx == j + 2)), j, :] = obj_features[(obj_idx == j) + (obj_idx == j + 1) + (obj_idx == j + 2)]
                idx[:torch.sum((obj_idx == j) + (obj_idx == j + 1) + (obj_idx == j + 2)), j] = obj_idx[(obj_idx == j) + (obj_idx == j + 1) + (obj_idx == j + 2)]
                
                position_embed[:torch.sum(obj_idx == j), j, :] = self.position_embedding.weight[0]
                position_embed[torch.sum(obj_idx == j):torch.sum((obj_idx == j) + (obj_idx == j + 1)), j, :] = self.position_embedding.weight[1]
                position_embed[torch.sum((obj_idx == j) + (obj_idx == j + 1)):torch.sum((obj_idx == j) + (obj_idx == j + 1) + (obj_idx == j + 2)), j, :] = self.position_embedding.weight[2]
                
            obj_input = obj_input + position_embed
            masks = (torch.sum(obj_input.view(-1, obj_features.shape[1]),dim=1) == 0).view(l, b - 2).permute(1, 0)
            
            obj_output = self.self_attention(obj_input, masks)
            
            output = torch.zeros_like(obj_features)
            for j in range(b - 2):
                if j == 0:
                    output[obj_idx == j] = obj_output[:, j][idx[:, j] == j]
                if j == b - 3:
                    output[obj_idx == j + 2] = obj_output[:, j][idx[:, j] == j + 2]
                
                output[obj_idx == j + 1] = obj_output[:, j][idx[:, j] == j + 1]
        else:
            l = obj_idx.shape[0]
            
            obj_input = obj_features.unsqueeze(1)
            
            if b == 2:
                position_embed = torch.zeros([l, 1, obj_features.shape[1]]).to(obj_features.device)
                position_embed[:torch.sum(obj_idx == 0), 0, :] = self.position_embedding.weight[0]
                position_embed[torch.sum(obj_idx == 0): torch.sum((obj_idx == 0) + (obj_idx == 1)), 0, :] = self.position_embedding.weight[1]
                obj_input = obj_input + position_embed
            
            masks = (torch.sum(obj_input.view(-1, obj_features.shape[1]),dim=1) == 0).view(l, 1).permute(1, 0)
            
            obj_output = self.self_attention(obj_input, masks)
            
            output = torch.zeros_like(obj_features)
            output[0] = obj_output[:, 0][obj_idx == 0]
            if b == 2:
                output[1] = obj_output[:, 0][obj_idx == 1]
        
        entry['obj_features'] = output
        
        return entry
    
