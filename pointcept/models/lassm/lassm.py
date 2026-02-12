"""
LaSSM.

Author: Lei Yao (rayyohhust@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from pointcept.custom.misc import process_label, \
    process_instance, split_offset
from pointcept.models.utils.structure import Point

from ..builder import MODELS, build_model
from .utils import mask_matrix_nms


@MODELS.register_module("LaSSM")
class LaSSM(nn.Module):
    def __init__(
        self,
        backbone,
        decoder=None,
        criterion=None,
        semantic_num_classes=18,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        topk_insts=100,
        score_thr=0.0,
        npoint_thr=100,
        sp_score_thr=0.4,
        nms=False,
        normliaze=True,
        freeze_backbone=False,
    ):
        super().__init__()

        # ignore info
        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        # backbone and pooling
        self.backbone = build_model(backbone)
        # decoder
        self.decoder = build_model(decoder)

        self.criterion = build_model(criterion)

        self.topk_insts = topk_insts
        self.score_thr = score_thr
        self.npoint_thr = npoint_thr
        self.sp_score_thr = sp_score_thr
        self.nms = nms
        self.normliaze = normliaze
        self.backbone_type = backbone["type"]
        
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        pt_offset = input_dict["origin_offset"].int()
        vx_offset = input_dict["offset"].int()

        assert "PT" or "SpUNet" in self.backbone_type
        if "PT" in self.backbone_type:
            point = Point(input_dict)
            point = self.backbone(point)
            feats = point.feat
        elif "SpUNet" or "Res16UNet34C" in self.backbone_type:
            feats = self.backbone(input_dict)
        input_dict["feats"] = feats
        
        assert "inverse" in input_dict.keys()
        inv = split_offset(input_dict["inverse"], pt_offset)
        sp = split_offset(input_dict["superpoint"], pt_offset)
        sp = [torch.unique(_sp, return_inverse=True)[1] for _sp in sp]
        vx_feat = split_offset(feats, vx_offset)

        sp_feat = [scatter_mean(_f[_i], _s, dim=0) 
                   for _f, _i, _s in zip(vx_feat, inv, sp)]
        
        input_dict["sp"] = sp
        input_dict["sp_feat"] = sp_feat
        input_dict["vx_feat"] = vx_feat
        input_dict["vx_coord"] = split_offset(input_dict["coord"], vx_offset)
        input_dict["inv"] = inv

        out = self.decoder(input_dict)

        # prepare target
        if ("segment" and "instance") in input_dict.keys():
            assert "select_idx" in out.keys()
            target = self.prepare_target(input_dict, out["select_idx"])
            return_dict = self.criterion(out, target)
        else:
            return_dict = {}

        if not self.training:
            return_dict = self.prediction(out, return_dict, sp)
        return return_dict
    
    def prepare_target(self, input_dict, select_idx=None):
        pt_offset = input_dict["origin_offset"].int()

        pt_ins = split_offset(input_dict["origin_instance"], pt_offset)
        pt_sem = split_offset(input_dict["origin_segment"], pt_offset)
        pt_coord = split_offset(input_dict["origin_coord"], pt_offset)
        sp = input_dict["sp"]

        target = dict()
        target["inst_gt"] = []
        target["vx_gt"] = dict()
        target["sp_cls"] = []

        vx_sem = process_label(
            input_dict["segment"].clone(), self.segment_ignore_index
        )
        vx_sem[vx_sem == self.semantic_ignore_index] = self.semantic_num_classes
        target["vx_gt"]["labels"] = vx_sem

        for p_ins, p_cls, p_sp, p_coord in zip(pt_ins, pt_sem, sp, pt_coord):
            # get instance mask
            p_ins = process_instance(
                p_ins.clone(), p_cls.clone(), self.segment_ignore_index
            )
            # get class mask
            p_cls = process_label(
                p_cls.clone(), self.segment_ignore_index
            )

            p_sem_mask = p_cls.clone()
            p_sem_mask[p_sem_mask == self.semantic_ignore_index] = self.semantic_num_classes
            p_sem_mask = torch.nn.functional.one_hot(p_sem_mask)
            p_sem_mask = p_sem_mask.T
            sp_sem_mask = scatter_mean(p_sem_mask.float(), p_sp, dim=-1)
            sp_sem_mask = sp_sem_mask > 0.5
            target["sp_cls"].append(sp_sem_mask)
                        
            # create gt instance markup     
            p_ins_mask = p_ins.clone()
            
            if torch.sum(p_ins_mask ==  self.instance_ignore_index) != 0:
                p_ins_mask[p_ins_mask ==  self.instance_ignore_index] \
                    = torch.max(p_ins_mask) + 1
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)[:, :-1]
            else:
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)

            if p_ins_mask.shape[1] != 0:
                p_ins_mask = p_ins_mask.T
                sp_ins_mask = scatter_mean(p_ins_mask.float(), p_sp, dim=-1)
                sp_ins_mask = sp_ins_mask > 0.5
            else:
                sp_ins_mask = p_ins_mask.new_zeros(
                    (0, p_sp.max() + 1), dtype=torch.bool)

            insts = p_ins.unique()[1:]

            scene_min, scene_max = p_coord.min(0)[0], p_coord.max(0)[0]
            gt_labels = insts.new_zeros(len(insts))
            
            gt_bboxes = []
            for inst in insts:
                index = p_cls[p_ins == inst][0]
                gt_labels[inst] = index

                inst_coord = p_coord[p_ins == inst]
                bbox_max = inst_coord.max(0)[0]
                bbox_min = inst_coord.min(0)[0]

                bbox_center_unnorm = (bbox_max + bbox_min) / 2
                if self.normliaze:
                    bbox_center = (bbox_center_unnorm - scene_min) / (scene_max - scene_min)
                else:
                    bbox_center = bbox_center_unnorm
                gt_bboxes.append(bbox_center)
            
            if len(gt_bboxes) != 0:
                gt_bboxes = torch.stack(gt_bboxes)
            else:
                gt_bboxes = p_coord.new_zeros((0, 3))

            target["inst_gt"].append(
                dict(labels=gt_labels, masks=sp_ins_mask, bboxes=gt_bboxes)
            )
            
        if select_idx is not None:
            for i in range(len(select_idx)):
                if target["inst_gt"][i]["masks"].shape[0] != 0:
                    target["inst_gt"][i]["query_masks"] = \
                        target["inst_gt"][i]["masks"][:, select_idx[i]]
                else:
                    target["inst_gt"][i]["query_masks"] = \
                        target["inst_gt"][i]["masks"]
        return target
    
    def prediction(self, out, return_dict, sp=None):
        scores, masks, classes = self.predict_by_feat(out, sp)
        masks = masks.cpu().detach().numpy()
        classes = classes.cpu().detach().numpy()

        sort_scores = scores.sort(descending=True)
        sort_scores_index = sort_scores.indices.cpu().detach().numpy()
        sort_scores_values = sort_scores.values.cpu().detach().numpy()
        sort_classes = classes[sort_scores_index]
        sorted_masks = masks[sort_scores_index]

        return_dict["pred_scores"] = sort_scores_values
        return_dict["pred_masks"] = sorted_masks
        return_dict["pred_classes"] = sort_classes  
        return_dict["select_idx"] = out["select_idx"]

        return return_dict

    def predict_by_feat(self, out, sp=None):
        cls_preds = out["labels"][0]
        pred_masks = out["masks"][0]
        
        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out["scores"] is not None:
            scores *= out["scores"][0]
        labels = torch.arange(
            self.semantic_num_classes, 
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.semantic_num_classes,
                             rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
        scores = scores * mask_scores

        if self.nms:
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                    mask_pred_sigmoid, labels, scores, kernel="linear")

        if sp is not None:
            mask_pred_sigmoid = mask_pred_sigmoid[:, sp[0]]
        mask_pred = mask_pred_sigmoid > self.sp_score_thr

        # score_thr
        score_mask = scores > self.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return scores, mask_pred, labels
    
