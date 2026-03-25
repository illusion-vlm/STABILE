import os
import cv2
import copy
import time
import torch
import random
import datetime
from cv2 import imread
import numpy as np
np.set_printoptions(precision=4)

from lib.config import Config
from lib.logger import logger
from lib.stabile import stabile
from lib.object_detector import detector
from dataloader.action_genome import AG, cuda_collate_fn
from lib.evaluation_recall import BasicSceneGraphEvaluator

conf = Config()

assert conf.mode in ('sgdet', 'sgcls', 'predcls')
assert conf.contrastive_type in ('uml', 'linear')

for i in conf.args:
    print(i,':', conf.args[i])

save_path = os.path.join('output', conf.mode)
log_val_path = os.path.join(save_path, 'log_val.txt')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
conf.data_path = os.path.join(conf.base_path, conf.data_path)

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path,
                filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')

object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode,
                           c_path=os.path.join(conf.base_path, 'models/stabile/faster_rcnn_ag.pth')).to(device=gpu_device)
object_detector.eval()

model = stabile(mode=conf.mode, contrastive_type=conf.contrastive_type,
                g_path=os.path.join(conf.base_path, 'glove'),
                attention_class_num=len(AG_dataset.attention_relationships), 
                spatial_class_num=len(AG_dataset.spatial_relationships),
                contact_class_num=len(AG_dataset.contacting_relationships), 
                obj_classes=AG_dataset.object_classes, 
                obj_att_layer_num=conf.obj_att_layer,
                enc_layer_num=conf.enc_layer, 
                dec_layer_num=conf.dec_layer, 
                obj_retriever=conf.obj_retriever).to(device=gpu_device)
model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)

print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))

evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    output_dir=save_path,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    output_dir=save_path,
    iou_threshold=0.5,
    constraint='no')

start_time = time.time()

with torch.no_grad():
    for b, data in enumerate(dataloader):
        
        if data[6] % 100 == 0 and data[6] > 0:
            print('index: ',data[6], flush=True)
        
        im_data = copy.deepcopy(data[0].to(device=gpu_device))
        im_info = copy.deepcopy(data[1].to(device=gpu_device))
        gt_boxes = copy.deepcopy(data[2].to(device=gpu_device))
        num_boxes = copy.deepcopy(data[3].to(device=gpu_device))
        in_idx = copy.deepcopy(data[4].to(device=gpu_device))
        audio_embeddings = copy.deepcopy(data[5].to(device=gpu_device))
        gt_annotation = AG_dataset.gt_annotations[data[6]]
        frame_names = AG_dataset.video_list[data[6]]
        
        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        entry['audio_features'] = audio_embeddings
        pred = model(entry)
        
        evaluator1.evaluate_scene_graph(gt_annotation, dict(pred))
        evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
str_print = 'Inference time {}'.format(total_time_str)
print(str_print, flush=True)

logger(log_val_path, str_print + '\n')

print('-------------------------with constraint-------------------------------')
evaluator1.print_stats(log_file_path=log_val_path)
print('-------------------------no constraint-------------------------------')
evaluator2.print_stats(log_file_path=log_val_path)

