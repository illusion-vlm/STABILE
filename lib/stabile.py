"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes

from lib.transformer import transformer
from lib.object_retriever import ObjectRetriever
from lib.relationship_classifier import RelationshipClassifier


class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet',
                 obj_classes=None,
                 obj_att_layer_num=None,
                 obj_retriever=None,
                 g_path=None):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode
        self.is_obj_retriever = obj_retriever
        

        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img = 64
        self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        if mode == 'sgcls':
            embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir=g_path, wv_dim=200)
            self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
            self.obj_embed.weight.data = embed_vecs.clone()
        elif mode == 'sgdet':
            embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir=g_path, wv_dim=200)
            self.obj_embed = nn.Embedding(len(obj_classes), 200)
            self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        
        self.obj_dim = 2048 + 200 + 128
        self.img_dim = 2048
        
        if self.is_obj_retriever:
            self.object_retriever = ObjectRetriever(img_dim=self.img_dim,
                                                 obj_dim=self.obj_dim,
                                                 att_nhead=8,
                                                 att_dim_feedforward=2048,
                                                 att_dropout=0.1,
                                                 att_layer_num=obj_att_layer_num,
                                                 sta_obj_classes_num=len(self.classes)-1)
            if mode == 'sgcls':
                self.decoder_lin = nn.Linear(self.obj_dim, len(self.classes)-1)
            else:
                self.decoder_lin = nn.Linear(self.obj_dim, len(self.classes))
        else:
            self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim, 1024),
                                             nn.BatchNorm1d(1024),
                                             nn.ReLU(),
                                             nn.Linear(1024, len(self.classes)))

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).to(scores.device)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        return entry

    def forward(self, entry):
        
        if self.mode != 'predcls' and self.training:
            # create object gt labels for frames
            FINAL_LABELS = entry['labels']
            FINAL_OBJ_IDX = entry['boxes'][:, 0].long()
            b = int(FINAL_OBJ_IDX[-1]+1)
            FINAL_IMG_LABELS = torch.zeros([b, len(self.classes)], dtype=torch.int64).to(FINAL_OBJ_IDX.device)
            for i in range(b):
                FINAL_IMG_LABELS[i][FINAL_LABELS[FINAL_OBJ_IDX==i]] = 1
            entry['img_labels'] = FINAL_IMG_LABELS[:,1:]

        if self.mode  == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        elif self.mode == 'sgcls':

            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            entry['obj_features'] = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            
            if self.is_obj_retriever:
                entry = self.object_retriever(entry)
            
            entry['distribution'] = self.decoder_lin(entry['obj_features'])
            
            if self.training:
                entry['pred_labels'] = entry['labels']
                entry['labels'] = entry['labels'] - 1 # for calculating loss since "background" object is no longer considered when calculating distribution
            else:
                entry['distribution'] = torch.softmax(entry['distribution'], dim=1)
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                box_idx = entry['boxes'][:,0].long()
                b = int(box_idx[-1] + 1)

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] ==duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class

                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class - 1])[:-1]
                        for j in ppp:

                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class-1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])+1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])


                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]: # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['base_features'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
            return entry
        else:
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            entry['obj_features'] = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            
            if self.is_obj_retriever:
                entry = self.object_retriever(entry)
            
            entry['distribution'] = self.decoder_lin(entry['obj_features'])
            
            if self.training:
                entry['pred_labels'] = entry['labels']
            else:
                entry['distribution'] = torch.softmax(entry['distribution'], dim=1)[:, 1:]
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2
                
                box_idx = entry['boxes'][:, 0].long()
                b = int(box_idx[-1] + 1)

                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                for i in range(b):
                    # images in the batch
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]

                    for j in range(len(self.classes) - 1):
                        # NMS according to obj categories
                        inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_dists = scores[inds]
                            cls_feats = feats[inds]
                            cls_scores = cls_dists[:, j]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds]
                            cls_dists = cls_dists[order]
                            cls_feats = cls_feats[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                            final_dists.append(cls_dists[keep.view(-1).long()])
                            final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                                        1).to(scores.device),
                                                          cls_boxes[order, :][keep.view(-1).long()]), 1))
                            final_feats.append(cls_feats[keep.view(-1).long()])

                entry['boxes'] = torch.cat(final_boxes, dim=0)
                box_idx = entry['boxes'][:, 0].long()
                entry['distribution'] = torch.cat(final_dists, dim=0)
                entry['features'] = torch.cat(final_feats, dim=0)
                
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][
                                                       box_idx == i, 0])  # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['base_features'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]), 
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)

            return entry
    

class stabile(nn.Module):

    def __init__(self, mode='sgdet',
                 contrastive_type=None,
                 g_path=None,
                 attention_class_num=None,
                 spatial_class_num=None,
                 contact_class_num=None,
                 obj_classes=None,
                 rel_classes=None,
                 obj_att_layer_num=None,
                 enc_layer_num=None,
                 dec_layer_num=None,
                 obj_retriever=None,
                 s_K=None,
                 c_K=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(stabile, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.a_class_num = attention_class_num
        self.s_class_num = spatial_class_num
        self.c_class_num = contact_class_num
        self.contrastive_type = contrastive_type
        self.mode = mode

        self.object_classifier = ObjectClassifier(mode=self.mode,
                                                  obj_classes=self.obj_classes,
                                                  obj_att_layer_num=obj_att_layer_num,
                                                  obj_retriever=obj_retriever,
                                                  g_path=g_path)

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )

        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
            
        self.vr_fc = nn.Linear(256*7*7, 512)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir=g_path, wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, 
                                              embed_dim=1936, nhead=8, dim_feedforward=2048, dropout=0.1, mode='latter')

        if self.contrastive_type == 'linear':
            self.a_rel_compress = nn.Linear(1936, self.a_class_num)
            self.s_rel_compress = nn.Linear(1936, self.s_class_num)
            self.c_rel_compress = nn.Linear(1936, self.c_class_num)
        else:
            self.a_rel_compress = RelationshipClassifier(embed_dim=1936, dropout=0.1, num_classes=self.a_class_num, contrastive=False)
            self.s_rel_compress = RelationshipClassifier(embed_dim=1936, dropout=0.1, num_classes=self.s_class_num, K=s_K)
            self.c_rel_compress = RelationshipClassifier(embed_dim=1936, dropout=0.1, num_classes=self.c_class_num, K=c_K)
        
        
    def set_cls_weight(self, s_cls_weight, c_cls_weight, base_K=3):

        self.s_rel_compress.set_cls_weight(s_cls_weight, base_K)
        self.c_rel_compress.set_cls_weight(c_cls_weight, base_K)
        

    def forward(self, entry):

        entry = self.object_classifier(entry)

        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)
        
        # Spatial-Temporal Transformer
        global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'])
        
        bs = global_output.shape[0]
        
        if self.contrastive_type == 'linear':
            entry["a_logits"] = self.a_rel_compress(global_output)
            entry["s_logits"] = self.s_rel_compress(global_output)
            entry["c_logits"] = self.c_rel_compress(global_output)
        else:
            entry["a_logits"], entry['a_feats'] = self.a_rel_compress(global_output, bs)
            if self.training:
                global_output = torch.cat([global_output, global_output, global_output], dim=0)
                
                s_labels_anchor = torch.tensor(entry['main_spatial_gt'], dtype=torch.long).to(global_output.device)
                s_logits, s_feats_anchor, s_feats_sample, s_targets_sample, s_idx_sample = self.s_rel_compress(global_output, bs, s_labels_anchor, entry['in_idx'])
                entry['s_logits'] = s_logits
                entry['s_feats_anchor'] = s_feats_anchor
                entry['s_feats_sample'] = s_feats_sample
                entry['s_targets_sample'] = s_targets_sample
                entry['s_idx_sample'] = s_idx_sample
                
                c_labels_anchor = torch.tensor(entry['main_contacting_gt'], dtype=torch.long).to(global_output.device)
                c_logits, c_feats_anchor, c_feats_sample, c_targets_sample, c_idx_sample = self.c_rel_compress(global_output, bs, c_labels_anchor, entry['in_idx'])
                entry['c_logits'] = c_logits
                entry['c_feats_anchor'] = c_feats_anchor
                entry['c_feats_sample'] = c_feats_sample
                entry['c_targets_sample'] = c_targets_sample
                entry['c_idx_sample'] = c_idx_sample
            else:
                entry["s_logits"], entry['s_feats'] = self.s_rel_compress(global_output, bs)
                entry["c_logits"], entry['c_feats'] = self.c_rel_compress(global_output, bs)
        
        return entry

