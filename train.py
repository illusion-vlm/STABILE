import os
import copy
import time
import datetime
import pandas as pd
import numpy as np
np.set_printoptions(precision=3)
from pytorch_metric_learning import losses as metric_loss

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.AdamW import AdamW
from lib.config import Config
from lib.logger import logger
from lib.stabile import stabile
from lib.object_detector import detector
from lib.losses import Logit_Compensation, UMLLoss
from dataloader.action_genome import AG, cuda_collate_fn
from lib.evaluation_recall import BasicSceneGraphEvaluator

"""------------------------------------some settings----------------------------------------"""
conf = Config()

assert conf.mode in ('sgdet', 'sgcls', 'predcls')
assert conf.scheduler_step in ('recall', 'mrecall')
assert conf.contrastive_type in ('uml', 'linear')

conf.save_path = os.path.join(conf.base_path, conf.save_path, conf.mode)
if conf.save_folder == 'datetime':
    conf.save_path = os.path.join(conf.save_path, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
else:
    conf.save_path = os.path.join(conf.save_path, conf.save_folder)
model_save_path = os.path.join(conf.save_path, 'models')
arg_file_path = os.path.join(conf.save_path, 'configurations.txt')
log_file_path = os.path.join(conf.save_path, 'logs.txt')
logval_file_path = os.path.join(conf.save_path, 'log_val.txt')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    
conf.data_path = os.path.join(conf.base_path, conf.data_path)

print('The CKPT saved here:', model_save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    str_print = '{} : {}'.format(i,conf.args[i])
    print(str_print, flush=True)
    logger(arg_file_path, str_print + '\n')
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, 
                      filter_nonperson_box_frame=True, filter_small_box=False if conf.mode=='predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4, collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, 
                     filter_nonperson_box_frame=True, filter_small_box=False if conf.mode=='predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4, collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")

# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode,
                           c_path=os.path.join(conf.base_path, 'models/stabile/faster_rcnn_ag.pth')).to(device=gpu_device)
object_detector.eval()

s_cls_weight = copy.deepcopy(AG_dataset_train.spat_cls_weight.to(device=gpu_device))
c_cls_weight = copy.deepcopy(AG_dataset_train.cont_cls_weight.to(device=gpu_device))

model = stabile(mode=conf.mode, contrastive_type=conf.contrastive_type,
                g_path=os.path.join(conf.base_path, 'glove'),
                attention_class_num=len(AG_dataset_train.attention_relationships), 
                spatial_class_num=len(AG_dataset_train.spatial_relationships),
                contact_class_num=len(AG_dataset_train.contacting_relationships), 
                obj_classes=AG_dataset_train.object_classes, 
                obj_att_layer_num=conf.obj_att_layer,
                enc_layer_num=conf.enc_layer, 
                dec_layer_num=conf.dec_layer, 
                obj_retriever=conf.obj_retriever,
                s_K=conf.s_k,
                c_K=conf.c_k).to(device=gpu_device)
if conf.contrastive_type == 'uml':
    model.set_cls_weight(s_cls_weight, c_cls_weight, conf.base_k)

evaluator =BasicSceneGraphEvaluator(mode=conf.mode, 
                                    AG_object_classes=AG_dataset_train.object_classes, 
                                    AG_all_predicates=AG_dataset_train.relationship_classes,
                                    AG_attention_predicates=AG_dataset_train.attention_relationships, 
                                    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                    AG_contacting_predicates=AG_dataset_train.contacting_relationships, 
                                    iou_threshold=0.5, constraint='with')
evaluator1 =BasicSceneGraphEvaluator(mode=conf.mode, 
                                     AG_object_classes=AG_dataset_train.object_classes, 
                                     AG_all_predicates=AG_dataset_train.relationship_classes,
                                     AG_attention_predicates=AG_dataset_train.attention_relationships, 
                                     AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                     AG_contacting_predicates=AG_dataset_train.contacting_relationships, 
                                     iou_threshold=0.5, constraint='no')
    
# loss function, default Multi-label margin loss
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()

if conf.contrastive_type == 'uml':
    s_uml_loss = UMLLoss(temperature=conf.losses_t, num_classes=len(AG_dataset_train.spatial_relationships),
                         cls_weight=s_cls_weight, sample_cls_count=model.s_rel_compress.K_per_cls).to(device=gpu_device)
    s_lc_loss = Logit_Compensation(cls_weight=s_cls_weight).to(device=gpu_device)
    c_uml_loss = UMLLoss(temperature=conf.losses_t, num_classes=len(AG_dataset_train.contacting_relationships),
                         cls_weight=c_cls_weight, sample_cls_count=model.c_rel_compress.K_per_cls).to(device=gpu_device)
    c_lc_loss = Logit_Compensation(cls_weight=c_cls_weight).to(device=gpu_device)

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=conf.factor, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

if not conf.no_logging:
    logger(log_file_path, '*'*60+'\n')
    logger(logval_file_path, '*'*60+'\n')
    
# some parameters
tr = []
best_recall = 0
best_Mrecall = 0

for epoch in range(conf.nepoch):
    model.train()
    object_detector.is_train = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    for b in range(len(dataloader_train)):
        data = next(train_iter)
        
        im_data = copy.deepcopy(data[0].to(device=gpu_device))
        im_info = copy.deepcopy(data[1].to(device=gpu_device))
        gt_boxes = copy.deepcopy(data[2].to(device=gpu_device))
        num_boxes = copy.deepcopy(data[3].to(device=gpu_device))
        in_idx = copy.deepcopy(data[4].to(device=gpu_device))
        # audio_embeddings = copy.deepcopy(data[5].to(device=gpu_device))
        gt_annotation = AG_dataset_train.gt_annotations[data[5]]
        
        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation ,im_all=None)
            # entry['audio_features'] = audio_embeddings
            entry['in_idx'] = in_idx[entry['im_idx'].long()]

        pred = model(entry)

        losses = {}
        
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])
            if conf.obj_retriever:
                losses['object_loss'] += F.multilabel_soft_margin_loss(pred['img_distribution_1'], pred['img_labels']) + \
                                         F.multilabel_soft_margin_loss(pred['img_distribution_2'], pred['img_labels'])
        
        a_targets = torch.tensor(pred["attention_gt"], dtype=torch.long).to(gpu_device).squeeze()
        losses["attention_relation_loss"] = ce_loss(pred['a_logits'], a_targets)
        
        s_main_targets_anchor = torch.tensor(pred['main_spatial_gt'], dtype=torch.long).to(device=gpu_device)
        c_main_targets_anchor = torch.tensor(pred['main_contacting_gt'], dtype=torch.long).to(device=gpu_device)
        
        # bce loss
        s_targets_anchor = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=gpu_device)
        c_targets_anchor = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=gpu_device)
        for i in range(len(pred["spatial_gt"])):
            s_targets_anchor[i, pred["spatial_gt"][i]] = 1
            c_targets_anchor[i, pred["contacting_gt"][i]] = 1
            
        if conf.contrastive_type == 'linear':
            losses["spatial_relation_loss"] = bce_loss(torch.sigmoid(pred['s_logits']), s_targets_anchor)
            losses["contact_relation_loss"] = bce_loss(torch.sigmoid(pred['c_logits']), c_targets_anchor)
        elif conf.contrastive_type == 'uml':
            losses["spatial_contrastive_relation_loss"] = (2.0 - conf.losses_alpha) * bce_loss(torch.sigmoid(pred['s_logits']), s_targets_anchor) + \
                                                            conf.losses_alpha * s_lc_loss(pred['s_logits'], s_targets_anchor) + \
                                                            conf.losses_beta * s_uml_loss(pred['s_feats_anchor'], s_targets_anchor, pred['s_feats_sample'], pred['s_targets_sample'])
            losses["contact_contrastive_relation_loss"] = (2.0 - conf.losses_alpha) * bce_loss(torch.sigmoid(pred['c_logits']), c_targets_anchor) + \
                                                            conf.losses_alpha * c_lc_loss(pred['c_logits'], c_targets_anchor) + \
                                                            conf.losses_beta * c_uml_loss(pred['c_feats_anchor'], c_targets_anchor, pred['c_feats_sample'], pred['c_targets_sample'])
        
        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % conf.log_iter == 0 and b >= conf.log_iter:
            time_per_batch = (time.time() - start) / conf.log_iter
            str_print = "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train), time_per_batch, len(dataloader_train) * time_per_batch / 60)
            print(str_print, flush=True)
            
            mn = pd.concat(tr[-conf.log_iter:], axis=1).mean(1)
            print(mn, flush=True)
            
            if not conf.no_logging:
                logger(log_file_path, str_print + '\n')
                for k in list(mn.keys()):
                    str_print = '{} : {:5f}'.format(k,mn[k])
                    logger(log_file_path, str_print + '\n')

            start = time.time()

    torch.save({"state_dict": model.state_dict()}, os.path.join(model_save_path, "model_{}.tar".format(epoch)))
    str_print = '*'*40 + '\n' + 'evaluate the checkpoint after {} epochs'.format(epoch)
    print(str_print)
    if not conf.no_logging:
        logger(log_file_path, str_print + '\n')

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in range(len(dataloader_test)):
            data = next(test_iter)
            
            im_data = copy.deepcopy(data[0].to(device=gpu_device))
            im_info = copy.deepcopy(data[1].to(device=gpu_device))
            gt_boxes = copy.deepcopy(data[2].to(device=gpu_device))
            num_boxes = copy.deepcopy(data[3].to(device=gpu_device))
            in_idx = copy.deepcopy(data[4].to(device=gpu_device))
            # audio_embeddings = copy.deepcopy(data[5].to(device=gpu_device))
            gt_annotation = AG_dataset_test.gt_annotations[data[5]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            # entry['audio_features'] = audio_embeddings
                
            pred = model(entry)
            evaluator.evaluate_scene_graph(gt_annotation, pred)
            evaluator1.evaluate_scene_graph(gt_annotation, pred)
        print('-----------', flush=True)
    
    if not conf.no_logging:
        logger(logval_file_path, 'epoch {} validation results:'.format(epoch) + '\n')
        evaluator.print_stats(logval_file_path)
        evaluator1.print_stats(logval_file_path)
        
    recall = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    if recall > best_recall:
        best_recall = recall
        str_print = 'new best recall of {} at epoch {}'.format(best_recall,epoch)
        print(str_print, flush=True)
        if not conf.no_logging:
            logger(logval_file_path, str_print + '\n')
            torch.save({"state_dict": model.state_dict()}, os.path.join(model_save_path, "best_recall_model.tar"))
    
    mrecall = evaluator.calc_mrecall()[20]
    if mrecall > best_Mrecall:
        best_Mrecall = mrecall
        str_print = 'new best Mrecall of {} at epoch {}'.format(best_Mrecall,epoch)
        print(str_print, flush=True)
        if not conf.no_logging:
            logger(logval_file_path, str_print + '\n')
            torch.save({"state_dict": model.state_dict()}, os.path.join(model_save_path, "best_Mrecall_model.tar"))
    
    evaluator.reset_result()
    evaluator1.reset_result()
    
    if conf.scheduler_step == 'recall':
        scheduler.step(recall)
    else:
        scheduler.step(mrecall)
