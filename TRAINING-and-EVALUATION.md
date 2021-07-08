## Training Steps

In this [notebook](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/blob/main/detr_custom_dataset.ipynb), you'll need to follow 4 steps in order to train the model

1. Prepare prerequisite
2. Setup Paths
3. Train Model

Note: If you want to edit and add new codes to this notebook, I suggest clone the repo or save the notebook as a copy or else you'll lose your new codes.

### Step 1 - Prepare Prerequisites 

1.1 Clone my github repository
```python
!git clone https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset.git
%cd Object-Detection-Using-DETR-CustomDataset/
```

1.2 Download face dataset from my github release page using wget
```bash
# Download train images
!wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/WIDER_train.zip -O datasets/train.zip

# Download val images
!wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/WIDER_val.zip -O datasets/val.zip

# Download annotations
!wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/wider_face_split.zip -O datasets/annotations.zip
```

1.3 There is an already implemented dataloader and you just need to copy face dataloader to detr/datasets folder. You can modify the dataloader if you want.
```bash
!cp dataloaders/face.py /content/Object-Detection-Using-DETR-CustomDataset/detr/datasets 

# Make folder for trained-weights.
!mkdir detr/outputs
```

1.4 Install COCO API for evaluation 
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

1.5 Dataset Prepartion 

There is a script [datasets/face_to_coco.py](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/blob/main/datasets/face_to_coco.py) which converts wider face dataset into COCO formatted dataset. Just simply run the cell and you will get train.json and val.json.

Before running the script, you need to set your dataset path in line 27 and 28 of the script.
```python
25.      phases = ["train", "val"]
26.      for phase in phases:
27.         root_path = "datasets/WIDER_{}/images/".format(phase)
28.         gt_path = os.path.join("datasets/wider_face_split/wider_face_{}.mat".format(phase))
```

1.6 After setting up, run the script **face_to_coco.py**

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
  |---- WIDER_test
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
```

### Step 3 - Train Model

To train the model, you'll need to run **detr/main.py** from with required arguments. If you want to see supported arguments and the definition of arguments, run below code.
```
!python detr/main.py --help
```

If you want to train model with my configurations, run the code below.

```
!python detr/main.py \
    --batch_size=16 \
    --epochs=15 \
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

## Evaluation Steps

To evaluate your model, you can use own trained weights or download my trained weights from github release page.

```bash
!wget https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases/download/v1.0/detr_r50_ep15.tar

# Extract tar file
!tar -xf detr_r50_ep15.tar
```

Load DETR model from torch.hub and load ckpt file into model

```python
TRAINED_CKPT_PATH = 'YOUR CHECKPOINT PATH' # sample --> checkpoint.pth
checkpoint = torch.load(TRAINED_CKPT, map_location='cpu')
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2)
model.load_state_dict(checkpoint['model'], strict=False)
```
For simplicity, I've modified postprocessing function from Standalone Colab Notebook of original detr repo.

```python
def postprocess_img(img_path): 
  im = Image.open(img_path)

  # mean-std normalize the input image (batch-size: 1)
  img = transform(im).unsqueeze(0)

  # propagate through the model
  start = time.time()
  outputs = model(img)
  end = time.time()
  print(f'Prediction time per image: {math.ceil(end - start)}s ', )

  # keep only predictions with 0.7+ confidence
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > 0.9

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  
  plot_results(im, probas[keep], bboxes_scaled)
```

Load paths from test folder
```python

TEST_IMG_PATH = 'datasets/WIDER_test/images'

img_format = {'jpg', 'png', 'jpeg'}
paths = list()

for obj in os.scandir(TEST_IMG_PATH):
  if obj.is_dir():
    paths_temp = [obj.path for obj in os.scandir(obj.path) if obj.name.split(".")[-1] in img_format]
    print()
    paths.extend(paths_temp)
```

Evaluate on first 10 images
```python
for i in paths[1:10]:
  postprocess_img(i)
```
## Metrics Visulaization 
After training detr, you will get output folder containing log.txt. By using that output folder, you can visualize your metrics. 
```
from detr.util.plot_utils import plot_logs
from pathlib import Path

log_directory = [Path('trained-weights')]

fields_of_interest = ('loss', 'mAP', 'loss_ce', 'loss_bbox', 'loss_giou', 'class_error', 'cardinality_error_unscaled')
plot_logs(log_directory, fields_of_interest)
```



