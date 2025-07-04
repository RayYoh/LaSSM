from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import VALID_CLASS_IDS_20, CLASS_LABELS_20
_base_ = ["../_base_/default_runtime.py"]

epoch = 512
eval_epoch = 512
# misc custom setting
batch_size = 12  # bs: total bs in all gpus
num_worker = 24
empty_cache = False
enable_amp = True
evaluate = True
if evaluate:
    evaluate_interval = [(1, 8), (496, 1)]
find_unused_parameters = False

class_names = CLASS_LABELS_20
class_ids = VALID_CLASS_IDS_20
num_classes = len(class_names)
segment_ignore_index = (-1, 0, 1)
semantic_num_classes = 18
num_channels = 32
normalize = True
weight = "exp/scannet/insseg-spformer-pt-spunet-v2/model/model_best.pth"

# model settings
model = dict(
    type="LaSSM",
    backbone=dict(
        type="SpUNet-v2m1",
        in_channels=6,
        num_channels=num_channels,
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True
    ),
    decoder=dict(
        type="LaSSMDecoder",
        num_class=semantic_num_classes,
        in_channel=num_channels,
        num_blocks=[1, 1, 1, 1, 1, 1],
        d_model=256,
        use_score=False,
        attn_mask=True,
        normliaze=normalize,
        alpha=0.8,
        num_query=400,
        k=8,
        t_layer=0,
        order=["hilbert", "hilbert-trans"],
    ),
    criterion=dict(
        type='InstanceCriterion',
        matcher=dict(
            type='HungarianMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='MaskBCECost', weight=1.0),
                dict(type='MaskDiceCost', weight=1.0),
                dict(type='BBoxCost', weight=1.0),]),
        loss_weight=[0.5, 1.0, 1.0, 1.0, 0.2],
        num_classes=semantic_num_classes,
        non_object_weight=0.1,
        fix_dice_loss_weight=False,
        iter_matcher=True,
        fix_mean_loss=True,),
    semantic_num_classes=semantic_num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    topk_insts=200,
    score_thr=0.0,
    npoint_thr=100,
    nms=True,
    normliaze=normalize,
)

# scheduler settings
optimizer = dict(type="AdamW", lr=0.0003, weight_decay=0.05)
scheduler = dict(type="PolyLR")

# dataset settings
dataset_type = "ScanNetSpDataset"
data_root = "data/scannet"

data = dict(
    # for the data, we need to load all categories
    num_classes=num_classes,
    ignore_label=-1,
    names=class_names,
    ids=class_ids,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="ToTensor"),
            dict(type="MeanShiftT"),
            dict(type="RandomDropoutT", dropout_ratio=0.2, dropout_application_ratio=0.5),
            dict(type="RandomFlipT", p=0.5),
            dict(type="RandomRotateT", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.95),
            dict(type="RandomRotateT", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotateT", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScaleT", scale=[0.8, 1.2]),
            dict(type="RandomTranslationT", shift=[0.1, 0.1, 0.1]),
            dict(type="CustomElasticDistortionT", distortion_params=[[10, 60], [30, 180]], p=0.5),
            dict(type="ChromaticAutoContrastT", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslationT", p=0.95, ratio=0.1),
            dict(type="ChromaticJitterT", p=0.95, std=0.05),
            dict(type="SphereCropT", point_max=250000, mode="random"),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "instance": "origin_instance",
                    "segment": "origin_segment",
                }
            ),
            dict(
                type="CustomGridSampleT",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
                keys=("coord", "color", "instance", "segment"),
            ),
            dict(type="NormalizeColorT"),
            dict(
                type="InstanceParserT", 
                segment_ignore_index=(-1, 0, 1),
                instance_ignore_index=-1
            ),
            dict(
                type="Collect",
                keys=(
                    "coord", 
                    "origin_coord",
                    "grid_coord", 
                    "segment", 
                    "origin_segment",
                    "instance", 
                    "origin_instance",
                    "instance_centroid",
                    "superpoint",
                    "inverse"
                ),
                feat_keys=("color", "coord"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            )
        ],
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="ToTensor"),
            dict(type="MeanShiftT"),
            dict(type="CustomElasticDistortionT", p=0.),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "instance": "origin_instance",
                    "segment": "origin_segment",
                }
            ),
            dict(
                type="CustomGridSampleT",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
                keys=("coord", "color", "instance", "segment"),
            ),
            dict(type="NormalizeColorT"),
            dict(
                type="InstanceParserT", 
                segment_ignore_index=(-1, 0, 1),
                instance_ignore_index=-1
            ),
            dict(
                type="Collect",
                keys=(
                    "coord", 
                    "origin_coord",
                    "grid_coord", 
                    "segment", 
                    "origin_segment",
                    "instance", 
                    "origin_instance",
                    "instance_centroid",
                    "superpoint",
                    "inverse"
                ),
                feat_keys=("color", "coord"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            )
        ],
        test_mode=False,
    ),
    test=dict(),
)

hooks = [
    dict(type="CustomCheckpointLoader", keywords="module.", replacement="module.", skip_key="decoder"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(
        type="CustomInformationWriter", 
        interval=50,
        key=("loss"),
    ),
    dict(
        type="SPInsEvaluator",
        segment_ignore_index=segment_ignore_index,
        semantic_ignore_index=(-1,),
        instance_ignore_index=-1,
    ),
    dict(type="CustomCheckpointSaver", save_freq=None),
]

# Tester
test = dict(type="InsSegTester", verbose=True)