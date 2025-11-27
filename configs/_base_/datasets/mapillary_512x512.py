mapillary_type = "CityscapesDataset"
mapillary_root = "/path/to/datasets/mapillary/"
mapillary_crop_size = (512, 512)
mapillary_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=mapillary_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
mapillary_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadStyleImageFromFile", style_folder="/path/to/StyleSet"), 
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    dict(type="StyleResize", scale=(2048, 1024)), 
    dict(type="ConcatStyle"), 
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_mapillary = dict(
    type=mapillary_type,
    data_root=mapillary_root,
    data_prefix=dict(
        img_path="training/images",
        seg_map_path="cityscapes_trainIdLabel/train/label",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=mapillary_train_pipeline,
)
val_mapillary = dict(
    type=mapillary_type,
    data_root=mapillary_root,
    data_prefix=dict(
        img_path="half/val_img",
        seg_map_path="half/val_label",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=mapillary_test_pipeline,
)
