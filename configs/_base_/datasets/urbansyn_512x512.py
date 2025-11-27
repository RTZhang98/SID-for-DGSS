urbansyn_type = "CityscapesDataset"
urbansyn_root = "/path/to/datasets/synthia"
urbansyn_crop_size = (512, 512)
urbansyn_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=urbansyn_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
urbansyn_train_pipeline_mask2former = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=urbansyn_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
urbansyn_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadStyleImageFromFile", style_folder="/path/to/StyleSet"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    dict(type="StyleResize", scale=(2048, 1024)), 
    dict(type="ConcatStyle"), 
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_urbansyn = dict(
    type=urbansyn_type,
    data_root=urbansyn_root,
    data_prefix=dict(
        img_path="RGB",
        seg_map_path="GT/LABELS_convert_v2",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=urbansyn_train_pipeline,
)
train_urbansyn_mask2former = dict(
    type=urbansyn_type,
    data_root=urbansyn_root,
    data_prefix=dict(
        img_path="RGB",
        seg_map_path="GT/LABELS_convert_v2",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=urbansyn_train_pipeline_mask2former,
)

val_urbansyn = dict(
    type=urbansyn_type,
    data_root=urbansyn_root,
    data_prefix=dict(
        img_path="RGB/test",
        seg_map_path="GT/LABELS_convert_v2/test",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=urbansyn_test_pipeline,
)
