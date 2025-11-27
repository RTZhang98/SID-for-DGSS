gta_type = "CityscapesDataset"
gta_root = "/path/to/datasets/gta/"
gta_crop_size = (512, 512)
gta_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=gta_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
gta_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadStyleImageFromFile", style_folder="/disk3/zhang_runtong/datasets/StyleSet"), 
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    dict(type="StyleResize", scale=(2048, 1024)), 
    dict(type="ConcatStyle"), 
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_train_pipeline,
)
val_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images/check",
        seg_map_path="labels/check",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_test_pipeline,
)