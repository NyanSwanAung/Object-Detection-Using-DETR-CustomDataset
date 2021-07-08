# Face Detection using DEtection TRansformers from Facebook AI 🚀
![PyTorch 1.5 +](https://img.shields.io/badge/Pytorch-1.5%2B-green)
![torch vision 0.6 +](https://img.shields.io/badge/torchvision%20-0.6%2B-green)

This repository includes 
* Training Pipeline for DETR on Custom dataset
* Wider Face Dataset annotaions and images
* Evaluation on test dataset 
* Trained weights for Wider Face Dataset in [release page](https://github.com/NyanSwanAung/Pothole-Detection-using-MaskRCNN/releases)
* Metrics Visualization 

## About Model 
DETR or DEtection TRansformer is Facebook’s newest addition to the market of available deep learning-based object detection solutions. Very simply, it utilizes the transformer architecture to generate predictions of objects and their position in an image. DETR is a joint Convolutional Neural Network (CNN) and Transformer with a feed-forward network as a head. This architecture allows the network to reliably reason about object relations in the image using the powerful multi-head attention mechanism inherent in the Transformer architecture using features extracted by the CNN.

![DETR Architecutre](https://miro.medium.com/max/1200/1*niV3pN0JvipfJeqmdWN-3g.png)

## Face Dataset

![Dataset Image](http://shuoyang1213.me/WIDERFACE/support/intro.jpg)

I've used [WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/) which is a publicly available face detection benchmark dataset, consisting of 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images. WIDER FACE dataset is organized based on 61 event classes. For each event class, the original dataset was split into 40%/10%/50% as training, validation and testing sets. 

By compiling the give code, the dataset will be automatically downloaded but you can download it manually from the official website or from my github [release page](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/releases). 

In [dataloader/face.py](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/blob/main/dataloaders/face.py), I set the maximum width of images in the random transform to 800 pixels. This should allow for training on most GPUs, but it is advisable to change back to the original 1333 if your GPU can handle it.

## Model 

We're going to use **DETR with a backbone of Resnet 50**, pretrained on COCO 2017 dataset. AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images, with torchscript transformer. If you want to use other DETR models, you can find them in model zoo below.

Model Zoo

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>

## Training and Evaluation Steps

Run all the cells of [detr_custom_dataset.ipynb](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/blob/main/DETR_custom_dataset.ipynb) to train your model without any errors in Google Colaboratory.

Follow this [readme](https://github.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/blob/main/TRAINING-and-INFERENCING.md) to understand the training pipeline of DETR and evaluation on test images.

## Results

![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/results1.png)
![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/results2.png)
![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/results3.png)
![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/results4.png)
![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/results5.png)


## COCO Evaluation Metrics on Validation Dataset (After 15 epochs of training)

It took me 4:59:45 hours to finish 15 epochs with batch_size=16 using Tesla P100-PCIE. If you want better accuracy, you can train more epochs.

```bash
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.766
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706
```
## Metrics Visualization 
![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/metrics1.png)
![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/metrics2.png)
![](https://raw.githubusercontent.com/NyanSwanAung/Object-Detection-Using-DETR-CustomDataset/main/assets/metrics3.png)

## Augmentation methods 
For train images, 
``` 
T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=800),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=800),
                ])
```

For val images, 

``` T.RandomResize([800], max_size=800) ```

## References

[DETR Tutorial by thedeepreader](https://github.com/thedeepreader/detr_tutorial)

[Training DETR on your own dataset by Oliver Gyldenberg Hjermitslev](https://towardsdatascience.com/training-detr-on-your-own-dataset-bcee0be05522)

[Facebook AI's original DETR repo](https://github.com/facebookresearch/detr)
