import os
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from lib.ConvGRU import ConvGRUCell

affine_par = True


class STANet(nn.Module):
    def __init__(self, obj_classes_num):
        super(STANet, self).__init__()
        all_channel = obj_classes_num # 35
        self.all_channel = all_channel
        self.extra_convs = nn.Conv2d(2048, all_channel, kernel_size=1)
        
        self.extra_conv_fusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=True)
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_gates_sta = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.extra_gates = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.extra_gateav = nn.Sequential(nn.Conv2d(all_channel*4, all_channel, 1),nn.ReLU(True),nn.Conv2d(all_channel, 1, 1))

        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)
        self.extra_lineara = nn.Linear(2048, all_channel*4)
        self.extra_linearv = nn.Linear(2048, all_channel*4)
        self.extra_refineST = nn.Sequential(
            nn.Conv3d(all_channel, all_channel, (3, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(all_channel, all_channel, (1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(all_channel, all_channel, (1, 3, 3), padding=(0, 1, 1)))
        self.extra_convv = nn.Conv2d(2048, all_channel*4, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, img_feat_top):
        
        bs = img_feat_top.size(0) # b = bs-2
        
        if bs > 2:
            x0s = img_feat_top[0:img_feat_top.size(0)-2]
            x1s = img_feat_top[1:img_feat_top.size(0)-1]
            x2s = img_feat_top[2:img_feat_top.size(0)]
            # a1 = aud_feat[1:aud_feat.size(0)-1]
            
            x0 = self.extra_convs(x0s) # (b,35,w,h)
            x0ss = F.avg_pool2d(x0, kernel_size=(x0.size(2), x0.size(3)), padding=0) # (b,35,1,1)
            x0ss = x0ss.view(-1, x0.size(1)) # (b,35)
            map_0 = x0 # (b,35,w,h)
            
            x1 = self.extra_convs(x1s) # (b,35,w,h)
            x1ss = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0) # (b,35,1,1)
            x1ss = x1ss.view(-1, x1.size(1)) # (b,35)
            map_1 = x1 # (b,35,w,h)
            
            x2 = self.extra_convs(x2s) # (b,35,w,h)
            x2ss = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0) # (b,35,1,1)
            x2ss = x2ss.view(-1, x2.size(1)) # (b,35)
            map_2 = x2 # (b,35,w,h)
            
            # av1 = self.AVfusion(a1, x1s) # (b,1,w,h)
            incat1 = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), 1).view(x1.size(0),x1.size(1),3,x1.size(2), x1.size(3)) # (b,35,3,w,h)
            # x11 = self.extra_conv_fusion(torch.cat((F.relu(x1+self.self_attention(x1)), F.relu(x1+x1*torch.sigmoid(self.extra_refineST(incat1).squeeze(2))), F.relu(x1+x1*av1)), 1)) # (b,35,w,h)
            x11 = self.extra_conv_fusion(torch.cat((F.relu(x1+self.self_attention(x1)), F.relu(x1+x1*torch.sigmoid(self.extra_refineST(incat1).squeeze(2)))), 1))
            x11 = self.extra_ConvGRU(x11, x1) # (b,35,w,h)
            map_all_1 = x11 # (b,35,w,h)
            x1sss = F.avg_pool2d(x11, kernel_size=(x11.size(2), x11.size(3)), padding=0) # (b,35,1,1)
            x1sss = x1sss.view(-1, x1.size(1)) # (b,35)
            
            _xss = torch.zeros([img_feat_top.size(0), x1ss.size(1)]).to(img_feat_top.device) # (bs,35)
            _xsss = torch.zeros([img_feat_top.size(0), x1ss.size(1)]).to(img_feat_top.device) # (bs,35)
            _map = torch.zeros([img_feat_top.size(0), map_1.size(1), map_1.size(2), map_1.size(3)]).to(img_feat_top.device) # (bs,35,w,h)
            
            _xss[0] = x0ss[0]
            _xss[1:img_feat_top.size(0)-1] = x1ss
            _xss[img_feat_top.size(0)-1] = x2ss[img_feat_top.size(0)-3]
            _xsss[0] = x0ss[0]
            _xsss[1:img_feat_top.size(0)-1] = x1sss
            _xsss[img_feat_top.size(0)-1] = x2ss[img_feat_top.size(0)-3]
            _map[0] = map_0[0]
            _map[1:img_feat_top.size(0)-1] = (map_1 + map_all_1) / 2
            _map[img_feat_top.size(0)-1] = map_2[img_feat_top.size(0)-3]
            
        else:
            _x = self.extra_convs(img_feat_top) # (bs,35,w,h)
            _xss = F.avg_pool2d(_x, kernel_size=(_x.size(2), _x.size(3)), padding=0) # (bs,35,1,1)
            _xss = _xss.view(-1, _x.size(1)) # (bs,35)
            _xsss = _xss # (bs,35)
            _map = _x # (bs,35,w,h)
        
        _sta = self.extra_gates_sta(_map) # (bs,1,w,h)
        _sta = _sta.view(_sta.size(0), -1) # (bs,w*h)
        _sta = F.softmax(_sta, dim=-1).view(_sta.size(0), -1, _sta.size(1)) # (bs,1,w*h)
        img_feat_top = img_feat_top.permute(0, 2, 3, 1) # (bs,w,h,2048)
        img_dim = img_feat_top.size(-1) # 2048
        img_feat_top = img_feat_top.view(img_feat_top.size(0), -1, img_dim) # (bs,w*h,2048)
        img_feat = torch.bmm(_sta, img_feat_top).view(-1, img_dim) # (bs,2048)
        
        return img_feat, _xss, _xsss, _map
        
    def self_attention(self, x):
        
        m_batchsize, C, width, height = x.size()
        f = self.extra_projf(x).view(m_batchsize, -1, width * height) # (b,35/2,w*h)
        g = self.extra_projg(x).view(m_batchsize, -1, width * height) # (b,35/2,w*h)
        h = self.extra_projh(x).view(m_batchsize, -1, width * height) # (b,35,w*h)
        
        attention = torch.bmm(f.permute(0, 2, 1), g) # (b,h*w,w*h)
        attention = F.softmax(attention, dim=1) # (b,w*h,w*h)

        self_attetion = torch.bmm(h, attention) # (b,35,w*h)
        self_attetion = self_attetion.view(m_batchsize, C, width, height) # (b,35,w,h)
        self_mask = self.extra_gates(self_attetion) # (b,1,w,h)
        self_mask = torch.sigmoid(self_mask) # (b,1,w,h)
        out = self_mask * x # (b,35,w,h)
        
        return out

    def AVfusion(self, audio, visual):
        
        bs, C, H, W = visual.shape
        visuals = self.extra_convv(visual) # (b,140,w,h)
        a_fea = self.extra_lineara(audio) # (b,140)
        a_fea = a_fea.view(bs, -1).unsqueeze(2) # (b,140,1)
        
        video_t= F.avg_pool2d(visual,kernel_size=(visual.size(2),visual.size(3)),padding=0) # (b,2048,1,1)
        video_t = video_t.view(bs, -1) # (b,2048)
        v_fea = self.extra_linearv(video_t).unsqueeze(1) # (b,1,140)
        
        att_wei = torch.bmm(a_fea, v_fea) # (b,140,140)
        att_wei = F.softmax(att_wei, dim=-1) # (b,140,140)
        att_v_fea = torch.bmm(att_wei, visuals.view(bs, self.all_channel*4, H*W)) # (b,140,w*h)
        att_v_fea = att_v_fea.view(bs, self.all_channel*4, H, W) # (b,140,w,h)
        self_mask = self.extra_gateav(att_v_fea) # (b,1,w,h)
        self_mask = torch.sigmoid(self_mask)# (b,1,w,h)
        
        return self_mask
        