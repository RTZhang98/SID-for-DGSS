cityscapes_type = "CityscapesDataset"
cityscapes_root = "/path/to/datasets/cityscapes/"
cityscapes_crop_size = (512, 512)
cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=cityscapes_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
cityscapes_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadStyleImageFromFile", style_folder="/path/to/StyleSet"), 
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    dict(type="StyleResize", scale=(2048, 1024)), 
    dict(type="ConcatStyle"), 
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/merge",
        seg_map_path="gtFine/merge",
    ),
    pipeline=cityscapes_train_pipeline,
)
val_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/val",
        seg_map_path="gtFine/val",
    ),
    pipeline=cityscapes_test_pipeline,
)
