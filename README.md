# RLB Object Detection

Based on [FastestDet by dog-qiuqiu](https://github.com/dog-qiuqiu/FastestDet/).

## Requirements

Run [train_env.sh](./train_env.sh) to create a Conda environment.

## Dataset

### Formatting
Dataset should be formatted like any other YOLO dataset:
```
.
├── train
│   ├── 000001.jpg
│   ├── 000001.txt
│   ├── 000002.jpg
│   ├── 000002.txt
│   ├── 000003.jpg
│   └── 000003.txt
└── val
    ├── 000043.jpg
    ├── 000043.txt
    ├── 000057.jpg
    ├── 000057.txt
    ├── 000070.jpg
    └── 000070.txt
```
Annotation files can contain multiple detections, one detection per line following the `class_id x y w h` format.

Example of a single annotation file:
```
11 0.344192634561 0.611 0.416430594901 0.262
14 0.509915014164 0.51 0.974504249292 0.972
```

### RLB dataset
- Download [RLB dataset](https://github.com/)
- Merge the content of `Augmented`, `Robot`, `Turret` and `Obstacle` folders into a `train` folder and the content of the `Test` folder into a `test` folder
- Move these folders into a `RLB_dataset` folder
- Move `RLB_dataset` into the `dataset` folder of this repo

### COCO2017 dataset
- Download COCO 2017 train and val images/annotations from [here](https://cocodataset.org/#download)
- Use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) to convert `instances_train2017.json` and `instances_val2017.json` to YOLO format
- Extract train images into a `train` folder, do the same for val images into a `test` folder
- Move these folders into a `COCO2017_dataset` folder
- Move `COCO2017_dataset` into the `dataset` folder of this repo

### Context augmentation
[https://www.kaggle.com/datasets/pankajkumar2002/random-image-sample-dataset](https://www.kaggle.com/datasets/pankajkumar2002/random-image-sample-dataset)