"""
LaSSM decoder.

Author: Lei Yao (rayyohhust@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import pointops
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS  
from torch_scatter import scatter_mean

from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
from .modeling_mamba import Mamba2Mixer
from .serialization import serialization
from fps_ops import point_sampler
    

class MambaAggregation(nn.Module):
    def __init__(self, d_model=256, dropout=0.0, k=8):
        super().__init__()
        self.k = k

        self.w_q = nn.Parameter(torch.empty(d_model, d_model))
        self.w_v = nn.Parameter(torch.empty(d_model, d_model))
        self.w_o = nn.Parameter(torch.empty(d_model, d_model))

        self.w_k = nn.Parameter(torch.empty(k, d_model))
        self.w_b = nn.Parameter(torch.empty(k))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.xavier_uniform_(self.w_o)

        nn.init.constant_(self.w_k, 0.0)
        nn.init.constant_(self.w_b, 0.0)
    
    def forward(self, query, query_pos, inst_feats, sp_coords):
        shortcut = query
        feat, _ = pointops.knn_query_and_group(
            inst_feats,
            sp_coords.float(),
            inst_feats.new_tensor([inst_feats.shape[0]]).int(),
            query_pos.float(),
            query.new_tensor([query.shape[0]]).int(),
            nsample=self.k
        )
        q, v = F.linear(query, self.w_q), F.linear(feat, self.w_v)
        q_expand = q.unsqueeze(1).expand(-1, self.k, -1)
        k = F.linear(q, self.w_k, self.w_b)
        k = F.softmax(k, dim=-1)
        w = torch.einsum('qk,qkd->qd', k, q_expand * v)
        output = self.dropout(F.linear(w, self.w_o)) + shortcut
        output = self.norm(output)
        return output
    

class MambaFFN(nn.Module):
    def __init__(self, d_model=256, hidden_dim=1024, dropout=0.0, activation_fn='gelu'):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, query):
        out = self.ffn(query) + query
        out = self.norm(out)
        return out


class SSM(nn.Module):
    def __init__(self, n_layer=2, d_model=256, order=["hilbert", "hilbert-trans"]):
        super().__init__()
        self.n_layer = n_layer
        self.order = order
        self.layers = nn.ModuleList([])

        Mamba = Mamba2Mixer
        _layer_cfg = Mamba2Config(
            hidden_size=d_model,
            state_size=64,
            n_groups=1,
            head_dim=64
        )

        for i in range(n_layer):
            layer = Mamba(_layer_cfg, layer_idx=i)
            self.layers.append(layer)
        self.pre_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, query_pos):
        for l in range(self.n_layer):
            shortcut = query
            query = self.pre_norm(query)
            query_token, origin_idx = [], []
            if query_pos.shape[0] <= 1:
                query_pos = torch.cat([query_pos, query_pos], dim=0)
                query = torch.cat([query, query], dim=0)
            _, order, inverse = serialization(query_pos, order=self.order)
            for i in range(len(self.order)):
                idx = order[i]
                query_sort = query[idx]
                query_token.append(query_sort.unsqueeze(0))
                o_idx = inverse[i]
                origin_idx.append(o_idx)
            query = torch.cat(query_token, dim=0)
            query = self.layers[l](query)
            query = torch.split(query, 1, dim=0)

            paths = []
            for i in range(len(self.order)):
                path = query[i].squeeze(0)[origin_idx[i]]
                paths.append(path)
            query = sum(paths) / len(paths)
            query = shortcut + query
            query = self.final_norm(query)
        return query
    

class MambaLayer(nn.Module):
    def __init__(
            self, d_model, n_layer=2, dropout=0., hidden_dim=1024, 
            activation_fn='gelu', k=8, order=["hilbert", "hilbert-trans"]
        ):
        super().__init__()
        self.aggregation = MambaAggregation(d_model, dropout, k)
        self.ssm = SSM(n_layer, d_model, order)
        self.ffn = MambaFFN(d_model, hidden_dim, dropout, activation_fn)

    def forward(self, queries, queries_pos, inst_feats, sp_coords, attn_mask=None):
        B = len(queries)
        out = []
        for i in range(B):
            query, query_pos = queries[i], queries_pos[i]
            assert not torch.isnan(query).any(), "query contains NaN values before aggregation"
            assert not torch.isnan(query_pos).any(), "query_pos contains NaN values before aggregation"
            inst_feat, sp_coord = inst_feats[i], sp_coords[i]

            query = self.aggregation(query, query_pos, inst_feat, sp_coord)
            assert not torch.isnan(query).any(), "query contains NaN values before SSM"
            assert not torch.isnan(query_pos).any(), "query_pos contains NaN values before SSM"
            query = self.ssm(query, query_pos)
            assert not torch.isnan(query).any(), "query contains NaN values before FFN"
            assert not torch.isnan(query_pos).any(), "query_pos contains NaN values before FFN"
            query = self.ffn(query)
            out.append(query)
        return out
    

class CrossMambaAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source, query, attn_mask=None):
        q, k, v = query, source, source
        output, _ = self.attn(q, k, v, attn_mask=attn_mask)
        output = self.dropout(output) + query
        output = self.norm(output)
        return output
    

class TransformerLayer(nn.Module):
    def __init__(
            self, d_model, n_layer=2, dropout=0., hidden_dim=1024, 
            activation_fn='gelu', order=["z","z-trans"]
        ):
        super().__init__()
        self.cross_attn = CrossMambaAttentionLayer(d_model, 8, dropout)
        self.ssm = SSM(n_layer, d_model, order)
        self.ffn = MambaFFN(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation_fn=activation_fn
        )

    def forward(self, queries, queries_pos, inst_feats, sp_coords, attn_masks=None):
        B = len(queries)
        out = []
        for i in range(B):
            query, query_pos = queries[i], queries_pos[i]
            inst_feat = inst_feats[i]
            attn_mask = attn_masks[i] if len(attn_masks) > 0 else None

            query = self.cross_attn(inst_feat, query, attn_mask)
            query = self.ssm(query, query_pos)
            query = self.ffn(query)
            out.append(query)
        return out


@MODELS.register_module("LaSSMDecoder")
class LaSSMDecoder(nn.Module):
    def __init__(
        self,
        num_blocks=[1, 1, 1, 1, 1, 1],
        num_class=18,
        in_channel=32,
        d_model=256,
        use_score=False,
        attn_mask=False,
        normliaze=True,
        alpha=0.8,
        num_query=400,
        k=8,
        t_layer=1,
        order=["hilbert", "hilbert-trans"]
    ):
        super().__init__()
        self.num_layers = len(num_blocks)
        self.use_score = use_score
        self.attn_mask = attn_mask
        self.num_class = num_class
        self.d_model = d_model
        self.normliaze = normliaze
        self.alpha = alpha
        self.num_query = num_query

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, num_blocks[i], order=order) if i < t_layer
                else MambaLayer(d_model, num_blocks[i], k=k, order=order)
                for i in range(self.num_layers)
            ]
        )

        self.input_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), 
            nn.LayerNorm(d_model), nn.ReLU())
        self.query_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), 
            nn.LayerNorm(d_model), nn.ReLU())
        self.sp_seg_head = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.ReLU(), 
            nn.Linear(d_model, num_class + 1))
        
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), 
            nn.Linear(d_model, num_class + 1))
        if self.use_score:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), 
                nn.Linear(d_model, 1))
        self.out_center = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), 
            nn.Linear(d_model, 3))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.ReLU(), 
            nn.Linear(d_model, d_model))
        
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _forward_head(self, query, mask_feats):
        pred_labels = []
        pred_masks, attn_masks = [], []
        pred_scores = [] if self.use_score else None
        pred_bias = []
        for i in range(len(query)):
            norm_query = self.out_norm(query[i])
            pred_labels.append(self.out_cls(norm_query))
            if self.use_score:
                pred_scores.append(self.out_score(norm_query))
            pred_bias.append(self.out_center(norm_query))
            pred_mask = torch.einsum('nd, md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return pred_labels, pred_scores, pred_masks, attn_masks, pred_bias
    
    def _get_query(self, input_dict, sp_seg):
        sp_feat = input_dict['sp_feat']   
        inv, sp = input_dict["inv"], input_dict["sp"]
        sel_idx = []

        sp_coord = [scatter_mean(_coord[_inv], _sp, dim=0)
                    for _coord, _inv, _sp in zip(input_dict["vx_coord"], inv, sp)]
        scene_range = [(x.min(0)[0], x.max(0)[0]) for x in sp_coord]
        sp_score = [x.softmax(dim=-1)[:, :-1] for x in sp_seg]
        
        sel_query, sel_query_pos = [], []
        for i, (_score) in enumerate(sp_score):
            num = int(self.alpha * _score.shape[0])
            if num > self.num_query:
                max_score, _ = _score.max(dim=-1)
                _, topk_idx = max_score.topk(num, sorted=True)

                sem_sp_feat = sp_feat[i][topk_idx, :]
                sem_sp_coord = sp_coord[i][topk_idx, :]

                fps_idx = point_sampler("d-fps", sem_sp_coord.unsqueeze(0),
                                        self.num_query).squeeze(0).long().unique()
                                        
                sel_query.append(self.query_proj(sem_sp_feat[fps_idx, :]))
                sel_query_pos.append(sem_sp_coord[fps_idx, :])
                sel_idx.append(topk_idx[fps_idx])
            else:
                sel_query.append(self.query_proj(sp_feat[i]))
                sel_query_pos.append(sp_coord[i])
                sel_idx.append(range(sp_feat[i].shape[0]))
        
        return sel_query, sel_query_pos, sel_idx, sp_coord, scene_range
    
    def forward(self, input_dict):
        sp_feat = input_dict['sp_feat']
        sp_seg = [self.sp_seg_head(x) for x in sp_feat]

        inst_feats = [self.input_proj(x) for x in sp_feat]
        mask_feats = [self.x_mask(x) for x in sp_feat]

        pred_labels, pred_masks, pred_scores, pred_centers = [], [], [], []

        query, query_pos, sel_idx, sp_coords, scene_range = self._get_query(input_dict, sp_seg)

        pred_label, pred_score, pred_mask, attn_mask, pred_bias = self._forward_head(query, mask_feats)
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        origin_query_pos = query_pos
        query_pos = [x + y for x, y in zip(origin_query_pos, pred_bias)]
        if self.normliaze:
            pred_center = [(x - y[0]) / (y[1] - y[0] + 1e-6) for x, y in zip(query_pos, scene_range)]
        else:
            pred_center = query_pos
        pred_centers.append(pred_center)

        for i in range(self.num_layers):
            query = self.layers[i](query, query_pos, inst_feats, sp_coords, attn_mask)
            pred_label, pred_score, pred_mask, attn_mask, pred_bias = self._forward_head(query, mask_feats)
            pred_labels.append(pred_label)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            origin_query_pos = query_pos
            query_pos = [x + y for x, y in zip(origin_query_pos, pred_bias)]
            if self.normliaze:
                pred_center = [(x - y[0]) / (y[1] - y[0] + 1e-6) for x, y in zip(query_pos, scene_range)]
            else:
                pred_center = query_pos
            pred_centers.append(pred_center)
        
        return {
            'labels': pred_label,
            'masks': pred_mask,
            'scores': pred_score,
            'bboxes': pred_center,
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'scores': c,
                'bboxes': d,
            } for a, b, c, d in zip(
                pred_labels[:-1],
                pred_masks[:-1],
                pred_scores[:-1],
                pred_centers[:-1]
            )],
            'sp_seg': sp_seg,
            'select_idx': sel_idx,
        }

