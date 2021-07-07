# Face Detection using DEtection TRansformers from Facebook AI
![PyTorch 1.5 +](https://img.shields.io/badge/Pytorch-1.5%2B-green)
![torch vision 0.6 +](https://img.shields.io/badge/torchvision%20-0.6%2B-green)


## Training Steps

### Step 1 - Prepare Prerequisites 

Clone my github repository
```python
!git clone https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset.git
%cd Object-Detection-Using-DETR-CustomDataset/
```

Download face dataset from my github release page using wget
```bash
# Download train images
!wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/WIDER_train.zip -O datasets/train.zip

# Download val images
!wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/WIDER_val.zip -O datasets/val.zip

# Download annotations
!wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/wider_face_split.zip -O datasets/annotations.zip
```

There is an implemented dataloader and you just need to copy face dataloader to detr/datasets folder. You can modify the dataloader if you want.
```bash
!cp dataloaders/face.py /content/Object-Detection-Using-DETR-CustomDataset/detr/datasets 
```

Download pretrained model [DETR	R50 from model zoo](https://github.com/facebookresearch/detr). 
Make folders for pre-trained model and trained weights.
``` bash
!mkdir detr/weights
!mkdir detr/outputs
!wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth -O detr/weights/detr-r50-e632da11.pth
```

Install COCO API for evaluation 
```bash
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

The original Wider Face Dataset contains images and .mat for annotations. In order to train DETR, we need the dataset to be in COCO format below like this.
```
--annotations  # annotation json files
  |---- train.json
  |---- val.json
--train # train images
  |---- image0
  |---- image1
  |---- ....
--val      # val images
  |---- image0
  |---- image1
  |---- ....
```

There is a script [datasets/face_to_coco.py](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/blob/main/datasets/face_to_coco.py) which converts wider face dataset into COCO formatted dataset. Just simply run the cell and you will get train.json and val.json.

Before running the script, you need to set your dataset path in line 27 and 28 of the script.
```python
25.      phases = ["train", "val"]
26.      for phase in phases:
27.         root_path = "datasets/WIDER_{}/images/".format(phase)
28.         gt_path = os.path.join("datasets/wider_face_split/wider_face_{}.mat".format(phase))
```

After setting up, run the script **face_to_coco.py**

```bash
!python datasets/face_to_coco.py

# Move json files to datasets folder
!mv train.json datasets && mv val.json datasets
```

After successful data preparation, you will get datasets folder like this
```
datasets
  |---- WIDER_train
  |---- WIDER_val
  |---- wider_face_split
  |---- train.json
  |---- val.json
```

### Step 2 - Setup Paths
```python
DATASET_PATH = 'datasets'

TRAIN_IMG = 'datasets/WIDER_train/images'
TRAIN_JSON = 'datasets/train.json'

VAL_IMG = 'datasets/WIDER_val/images'
VAL_JSON = 'datasets/val.json'

PRETRAINED_MODEL = 'detr/weights/detr-r50-e632da11.pth'
```

### Step 3 - Train Model

Run main.py to train the model. If you want to see optional arguments and the meaning of arguments, run the code below.
```
!python detr/main.py --help
```

If you want to train model with my configurations, run the code below.

```
!python detr/main.py \
    --batch_size=8 \
    --epochs=13 \
    --num_classes=2 \
    --num_queries=100 \
    --dataset_file='face' \
    --data_path={DATASET_PATH} \
    --train_folder={TRAIN_IMG} \
    --train_json={TRAIN_JSON} \
    --val_folder={VAL_IMG} \
    --val_json={VAL_JSON} \
    --output_dir='detr/outputs' \
    --resume={PRETRAINED_MODEL}
```
