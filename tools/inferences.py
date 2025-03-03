
import os
import torch

import numpy as np
import pyviz3d.visualizer as viz

from pointcept.models import build_model
from pointcept.engines.defaults import default_config_parser
from pointcept.custom.viz_scannetv2 import COLOR_DETECTRON2

from collections import OrderedDict
import pointcept.utils.comm as comm
from torch_scatter import scatter_mean
  

def read_data(path):
    print(f"Reading {path}")
    coord = np.load(os.path.join(path, "coord.npy"))
    color = np.load(os.path.join(path, "color.npy"))
    superpoint = np.load(os.path.join(path, "superpoint.npy"))

    sem_label = np.load(os.path.join(path, "segment20.npy"))
    ins_label = np.load(os.path.join(path, "instance.npy"))

    return coord, color, superpoint, sem_label, ins_label


def build_scenemamba(config_file=None, weight_path=None):
    cfg = default_config_parser(
        config_file, options=dict(weight=weight_path)
    )
    model = build_model(cfg.model)
    if os.path.isfile(cfg.weight):
        checkpoint = torch.load(cfg.weight)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
            weight[key] = value
        model.load_state_dict(weight, strict=True)
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(cfg.weight))
    return model


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def prediction(coord, color, superpoint, config_file=None, weight_path=None):
    device = torch.device("cuda:0")
    model = build_scenemamba(config_file, weight_path)
    model = model.to(device)
    model.eval()

    coord = torch.from_numpy(coord).float().to(device)
    color = torch.from_numpy(color).float().to(device)
    superpoint = torch.from_numpy(superpoint).long().to(device)
    input_dict = {
        "coord": coord,
        "color": color,
        "superpoint": superpoint,
    }
    device = input_dict["coord"].device

    # mean shift
    assert isinstance(input_dict["coord"], torch.Tensor)
    coord = input_dict["coord"]
    coord -= coord.mean(0)
    input_dict["coord"] = coord

    # elastic
    input_dict["elastic_coord"] = input_dict["coord"].cpu().numpy() / 0.02
    input_dict["elastic_coord"] = torch.tensor(input_dict["elastic_coord"]).to(device)

    input_dict["origin_coord"] = input_dict["coord"]

    # grid
    scaled_coord = input_dict["elastic_coord"] - input_dict["elastic_coord"].min(0)[0]
    grid_coord = torch.floor(scaled_coord).long()
    min_coord = grid_coord.min(0)[0]
    grid_coord -= min_coord
    scaled_coord -= min_coord
    min_coord = min_coord * scaled_coord.new_tensor([0.02])

    key = fnv_hash_vec(grid_coord.cpu().numpy())
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[0:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]
    idx_unique = torch.tensor(idx_unique, dtype=torch.long, device=device)

    inverse = torch.tensor(inverse, dtype=torch.long, device=device)
    idx_sort = torch.tensor(idx_sort, dtype=torch.long, device=device)
    input_dict["inverse"] = torch.zeros_like(inverse)
    input_dict["inverse"][idx_sort] = inverse
    input_dict["grid_coord"] = grid_coord[idx_unique]
    
    input_dict["coord"] = input_dict["coord"][idx_unique]
    input_dict["color"] = input_dict["color"][idx_unique]

    # normalize color
    input_dict["color"] = input_dict["color"] / 127.5 - 1

    input_dict["offset"] = torch.tensor([input_dict["coord"].shape[0]], dtype=torch.long, device=device)
    input_dict["origin_offset"] = torch.tensor([input_dict["origin_coord"].shape[0]], dtype=torch.long, device=device)
    input_dict["feat"] = torch.cat([input_dict["color"], input_dict["coord"]], dim=1)

    with torch.no_grad():
        pred = model(input_dict)
    select_idx = pred["select_idx"]
    idx = select_idx[0].cpu().numpy()
    return idx

    
if __name__ == "__main__":
    ours_config = "exp/scannet/paper/insseg-sm-spunet-v2-3-w2-0/config.py"
    ours_weight = "exp/scannet/paper/insseg-sm-spunet-v2-3-w2-0/model/model_best.pth"

    fps_config = "exp/scannet/paper/insseg-fps-sm-spunet-v2-3/config.py"
    fps_weight = "exp/scannet/paper/insseg-fps-sm-spunet-v2-3/model/model_best.pth"

    sem_config = "configs/scannet/insseg-sem-sm-spunet-v2-3.py"
    sem_weight = "exp/scannet/paper/insseg-sem-sm-spunet-v2-3/model/model_best.pth"


    coord, color, superpoint, sem_label, ins_label = read_data("data/scannet_sp/val/scene0196_00")
    ours_idx = prediction(coord, color, superpoint, ours_config, ours_weight)
    fps_idx = prediction(coord, color, superpoint, fps_config, fps_weight)
    sem_idx = prediction(coord, color, superpoint, sem_config, sem_weight)

    sp_coord = scatter_mean(torch.from_numpy(coord).float(), torch.from_numpy(superpoint).long(), dim=0)
    sp_coord = sp_coord.cpu().numpy()

    ours_sel_coord = sp_coord[ours_idx]
    fps_sel_coord = sp_coord[fps_idx]
    sem_sel_coord = sp_coord[sem_idx]

    ours_sel_color = np.ones_like(ours_sel_coord) * [0, 0, 139]
    fps_sel_color = np.ones_like(fps_sel_coord) * [0, 0, 139]
    sem_sel_color = np.ones_like(sem_sel_coord) * [0, 0, 139]

    mask_valid = (sem_label != -1)
    coord = coord[mask_valid]
    color = color[mask_valid]
    sem_label = sem_label[mask_valid]
    ins_label = ins_label[mask_valid]

    ins_label_rgb = np.zeros_like(color)
    mask = ~np.isin(sem_label, [-1, 0, 1])
    ins_label[~mask] = -1
    unique, inverse = np.unique(ins_label[mask], return_inverse=True)
    ins_label[mask] = inverse
    ins_label_rgb[~mask] = (200, 200, 200)
    for i, ind in enumerate(np.unique(inverse)):
        ins_label_rgb[ins_label == ind] = COLOR_DETECTRON2[i % 68]


    v = viz.Visualizer()
    v.add_points(f'input', coord, color, point_size=15.0)
    v.add_points(f'inst_gt', coord, ins_label_rgb, point_size=15.0)
    v.add_points(f'ours', ours_sel_coord, ours_sel_color, point_size=85.0)
    v.add_points(f'fps', fps_sel_coord, fps_sel_color, point_size=85.0)
    v.add_points(f'sem', sem_sel_coord, sem_sel_color, point_size=85.0)
    v.save(f"visualizations/pyviz3d")
